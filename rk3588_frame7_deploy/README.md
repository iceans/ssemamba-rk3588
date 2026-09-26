# RK3588 部署：frame7 TsgMamba 消融模型（DAUB）

目标权重：`runs/compare_result/ablation/Ablation/DAUB/frame7/weights/best.pt`

> ⚠️ 这与之前 `rk3588_deploy/` 里部署的 `yolov8n_813`（标准 YOLOv8，3 通道输入）
> **不是同一个模型**。`frame7` 是自定义 **TsgMamba** 结构，输入为 **10 通道时序立方体**，
> 含选择性扫描（SSM）与可变形卷积（DCN），部署难度完全不是一个量级。

---

## 1. 模型事实

| 项 | 值 |
|----|----|
| 结构 | `yolov8n_mamba.yaml` + DAUB 修改（TimeMambaStem / TMixSSM / SS2DguideMix / CCCBlock / DCNAlignment / Haar / SS2DTimeMix） |
| 输入 | `[1, 10, 640, 640]`：3 = 当前帧 RGB，7 = 前 6 帧 + 当前帧灰度（`frame_num=7`, `cube=True`） |
| 输出 | 6 个分离头：`box0/1/2 [1,64,h,w]` 原始 DFL logits，`cls0/1/2 [1,1,h,w]` 原始 cls logits |
| 类别 | 1 |
| 训练环境 | conda `mambayolo`（torch 2.0.0+cu118，CUDA 自定义 selective_scan 内核） |

### 为什么不能直接转 RKNN
1. **选择性扫描**：`tgsmamba.selective_scan_fn` 写死调用 `SelectiveScanCuda.apply`
   （CUDA 自定义内核），无法在 CPU/ONNX 追踪，且 RKNN 无对应算子。
2. **可变形卷积**：`torchvision.ops.DeformConv2d` 没有 ONNX symbolic，RKNN 也不支持。
3. **张量巨大**：`d_state=16`，在 stride-2 的 320×320 特征上扫描序列长度 `L≈5.1e5`，
   并行扫描中间张量 `(1,8,512000,16)` fp16 ≈ 131 MB/个。

## 2. 本目录做了什么

用**导出时非侵入式 monkeypatch**（不改动 `ultralytics/` 源码）把模型变成可导出：

| 原实现 | 替换 | 说明 |
|--------|------|------|
| `SelectiveScanCuda`（CUDA） | `pscan.selective_scan_parallel` | **精确**的 Hillis–Steele 结合律并行扫描，只用 Pad/Slice/Mul/Add/Exp/ReduceSum，可导出且 RKNN 友好 |
| `torchvision DeformConv2d` | `dcn_vect.deform_conv2d_forward` | 向量化 DCNv2：**一次 GridSample 采样全部 9 个核位置 + 一次 MatMul**，与原实现数值一致 |
| `einsum("b k d l, k c d -> b k c l")` | `MatMul` | 8 通道扫描投影，避免 Einsum 落到 RKNN CustomOperator |

另外修复了一个环境级 bug：仓库根目录曾有一个 **0 字节的 `onnx.py`**，会屏蔽真正的
`onnx` 包（`module 'onnx' has no attribute 'load_from_string'`），已删除。

## 3. 结果

- ONNX 导出成功：`onnx/frame7_640.onnx`（77 MB，~16 s）。
- RKNN fp16 转换成功：`model/frame7_640_fp16.rknn`（191 MB，~113 s）。
- 数值验证：
  - 并行扫描 vs 原 loop：`~3e-6`
  - DCN vs torchvision：`~5e-7`
  - ONNX-RT fp32 vs PyTorch fp32：相对误差 ≤ **0.8%**
  - RKNN fp16 vs PyTorch fp32（模拟器）：相对误差最大 **~36%（box0）/18%（cls0）** ⚠️

详见 `results/validation.txt`。

## 4. 已知风险（必须留意）

1. **DCN 是 CPU/CustomOperator**：RKNN 日志有
   `No lowering found for GridSample ... use CustomOperatorLower`，
   `rknn-toolkit-lite2` 在板端**未必能执行**。若不行，用 `export_onnx.py --dcn conv`
   退化为普通卷积（实测会让检测置信度从 0.55 掉到 0.47，需要重评精度）。
2. **fp16 精度损失大**：模拟器上 box0/cls0 相对误差达两位数百分比，长序列扫描在
   fp16 下动态范围不够。需要在板端用真实数据评估 mAP，必要时改为 int8 或对部分层保 fp32。
3. **速度/显存**：`Total Internal Memory` 约 4.2 GB（累加值，峰值未知），
   扫描是内存带宽瓶颈，端到端帧率很可能远低于实时。**RK3588 上大概率 < 2 FPS**。
4. **板子当前不在线**：`192.168.50.2` ping 不通（No route to host），未做板端实测。

## 5. 复现步骤

```bash
# 主机: 导出 ONNX (conda mambayolo 环境, 需 CUDA 加载权重)
cd rk3588_frame7_deploy/scripts
python export_onnx.py --imgsz 640 --out ../onnx/frame7_640.onnx          # 精确 DCN
# python export_onnx.py --imgsz 640 --dcn conv                          # 近似 DCN

# 主机: ONNX -> RKNN fp16 (conda rknn 环境, rknn-toolkit2 2.3.2)
python onnx2rknn_fp16.py --onnx ../onnx/frame7_640.onnx --out ../model/frame7_640_fp16.rknn
```

输入约定（对齐已有 `rk3588_deploy` 的经验）：
- fp16 模型：喂 **fp16 NHWC [0,255]**，runtime 内部 ÷255；输出 float32。
- 预处理/后处理与 `rk3588_deploy/scripts/eval_board.py` 一致（DFL softmax + sigmoid + NMS）。

## 6. 目录

```
rk3588_frame7_deploy/
├── scripts/
│   ├── pscan.py              # 并行选择性扫描（可导出）
│   ├── dcn_vect.py           # 向量化 DCNv2（GridSample+MatMul）
│   ├── export_onnx.py        # pt -> ONNX（打补丁）
│   └── onnx2rknn_fp16.py     # ONNX -> RKNN fp16
├── onnx/frame7_640.onnx
├── model/frame7_640_fp16.rknn
└── results/validation.txt
```

## 7. 板端实测结论（重要）

> 详见 `results/board_verification.txt`。板子：`ssh -o BatchMode=yes Tronlong@192.168.50.2`
> （driver 0.9.6，系统 `librknnrt.so` 2.0.0b0，NPU 驱动编入内核不可单独升级）。

| 测试 | 结果 |
|------|------|
| 板载旧 3ch YOLOv8n | 正常，59.8 ms |
| 最小 10 通道模型 | 正常，1.9 ms（说明 10 通道本身没问题）|
| frame7-640（faithful DCN） | `init_runtime` 段错误（`Unsupport CPU op: GridSample`）|
| frame7-640（conv DCN） | `init_runtime` 段错误（2.0.0b0）|
| frame7-192（未分块） | 推理在 op1 提交失败（扫描 Pad 产生 78848 维）|
| frame7-192（分块扫描） | 推进到 op122，`Mul` 提交失败 |
| librknnrt 2.3.2 覆盖系统库 | `job commit failed, ret:-22`（与 driver 0.9.6 不兼容）|

**结论：当前板端 NPU 栈下 frame7 无法运行。** 根因：
1. 2.3.2 运行时支持 GridSample 等，但需要更新的 NPU 内核驱动（本板驱动编入内核，需重编/烧写 BSP）；
2. 2.0.0b0 运行时不支持 GridSample，且对长序列扫描/大张量 `Mul`/`Slice` 提交失败；
3. 模型把 5 帧 × H × W 拉平成最长 512000 的序列做 SSM/BC 投影，超出 NPU 单张量维度上限（≈65535），
   且并行扫描展开成上百个超大算子（单 job 超 6 s 会被驱动超时 reset）。

`pscan.py` 已实现**分块并行扫描**（把长度维拆成 `(num_chunk, chunk)`，用 L 的因子避免补齐），
可消除扫描内部的超长维度，但跨扫描/BC 投影（`Concat/Gather/Transpose/Split`）仍需一并分块。

## 8. 待办 / 可选方向

- [ ] **A** 升级板端 BSP/NPU 驱动匹配 librknnrt 2.3.2（需 root/烧写），并做“全链路分块”。
- [ ] **B** 降分辨率（≤224）+ 全链路分块 + 普通卷积 DCN，先跑通低精度版本。
- [ ] **C** 为该边缘场景重设计轻量时序模型（替换 SSM、减少 `frame_num`）后重训/蒸馏。
- [ ] **D** 用 ONNX Runtime 在板端 CPU 跑 fp32（精度正确但慢；板端暂无 ort，需离线安装）。
- [ ] int8：需用真实的 10 通道立方体 `.npy` 校准集。
