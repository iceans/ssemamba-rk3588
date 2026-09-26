# RK3588 部署记录：yolov8n_813 (DAUB)

将 `runs/compare_result/DAUB/yolov8n_813/weights/best.pt` 部署到 RK3588 板子（Tronlong，`192.168.50.2`），
完成 **fp16** 与 **int8** 两种精度的转换、验证、FPS 基准与全量 DAUB 测试集评估。

## 1. 版本与环境

| 项 | 值 |
|----|----|
| 板子 | RK3588-Tronlong，Ubuntu 20.04.6 LTS，aarch64 |
| 板子 NPU 运行时 | `librknnrt.so` 2.0.0b0（driver 0.9.6）—— 可加载 2.3.2 模型（仅告警） |
| 板子推理 | miniconda `rknn` 环境，rknn-toolkit-lite2 2.3.2（numpy 2.0.2 / cv2 4.12） |
| 工作站转换 | conda `rknn` 环境，rknn-toolkit2 **2.3.2**（Python 3.9） |
| 工作站导出 | conda `mambayolo` 环境，torch 2.0.0+cu118 / ultralytics 8.3.234（补装 onnx 1.14.0） |

## 2. 转换链路

```
best.pt --(export_onnx_split.py)--> best_split.onnx
        --(onnx2rknn_fp16.py)--> best_fp16.rknn   (do_quantization=False)
        --(onnx2rknn_int8.py)--> best_int8.rknn   (asymmetric_quantized-8, 500张DAUB训练图校准)
```

- **输入**：`images` `[1,3,640,640]`，固定 640×640。
- **分离头输出（6 个）**：
  - `box0/1/2`：`[1,64,h,w]` 原始 DFL logits（软最大化在后处理）
  - `cls0/1/2`：`[1,1,h,w]` **原始 cls logits**（sigmoid 在后处理用 float32）
- **RKNN 配置**：`target_platform=rk3588`，`mean_values=[[0,0,0]]`，`std_values=[[255,255,255]]`。
- **输入约定（重要）**：
  - **fp16**：喂 **fp16 NHWC [0,255]**（runtime 自动 ÷255）。
  - **int8**：喂 **uint8 NHWC [0,255]**（runtime 自动 ÷255 + 量化）；rknnlite 输出自动反量化为 float32。

> cls 用原始 logit 而非模型内 sigmoid，是为了避免 fp16 sigmoid 下限（~0.00183）抬高背景框、
> 污染 mAP 低置信度尾部。后处理用 float32 sigmoid。

## 3. 后处理（YOLO 标准）

- DFL softmax → `(grid + 0.5 ± dist) * stride` → xyxy（640 空间）→ 按原图缩放。
- sigmoid(cls) → 置信度；单类 NMS。
- 评估：`conf>=0.001`、`IoU=0.7`（ultralytics val 默认）；检测展示可用 conf=0.5、IoU=0.45。

## 4. FPS（板子实测，100 次平均）

| 模式 | fp16 | int8 |
|------|:---:|:---:|
| 单核单图延迟 | ~50 ms → **~20 FPS** | ~24 ms → **~42 FPS** |
| 3 核合跑单模型 | ~20 FPS | ~44 FPS |
| 3 实例并行（吞吐） | ~52 FPS | **~113 FPS** |

## 5. 全量 DAUB 测试集指标（4795 张，YOLO/COCO 标准）

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|:---:|:---:|:---:|:---:|
| PyTorch fp32（参照） | 0.8605 | 0.7512 | 0.8128 | 0.5353 |
| RKNN-fp16（板子） | 0.8557 | 0.7591 | 0.7888 | 0.5133 |
| **RKNN-int8（板子）** | 0.8514 | 0.7527 | **0.7973** | **0.5206** |

- 全部 4795 张测试图均为 256×256，预处理与 ultralytics 一致。
- 单图逐元素验证：RKNN-fp16 与 PyTorch 主框差 **0.04px**；int8 主框差 **~0.3px**、4 框一致。
- **结论：int8 在精度和速度上均优于 fp16**（mAP50 +0.0085、mAP50-95 +0.0073，FPS 约 2.2×），
  且更接近 PyTorch 参照。**推荐部署 int8。**

## 6. 目录结构

```
rk3588_deploy/
├── README.md
├── model/
│   ├── best_fp16.rknn          # fp16 模型
│   └── best_int8.rknn          # int8 模型（推荐）
├── onnx/best_split.onnx
├── scripts/
│   ├── export_onnx_split.py    # pt -> 分离头 onnx
│   ├── onnx2rknn_fp16.py       # onnx -> rknn(fp16)
│   ├── onnx2rknn_int8.py       # onnx -> rknn(int8, 校准)
│   ├── eval_board.py           # 板子：fp16 全量评估
│   ├── eval_board_int8.py      # 板子：int8 全量评估
│   ├── bench_fp16.py / bench_int8.py   # FPS 基准
│   ├── compute_metrics.py / compute_metrics_int8.py  # 指标
│   └── verify_*.py             # 单图精度验证
└── results/
    ├── metrics.txt             # 三模型指标 + FPS 汇总
    ├── preds.json / preds_int8.json          # 板子预测
    ├── pt_out/onnx_out/rknn_out/int8_out.npz # 单图 6 输出对比
    └── calib_sample500.txt     # int8 校准清单（500 张）
```

## 7. 板子上现有文件（`/home/Tronlong/rknn_deploy/`）

- `best_fp16.rknn`、`best_int8.rknn`（两个部署模型）

## 8. 待办 / 后续

- **独立检测程序尚未编写**：目前只有批量评估脚本，没有接图片/视频/摄像头并画框输出的正式检测程序。
  现有板载 `/home/Tronlong/RK3588_uav/main.py+func.py` 是旧 int8 模型 + 旧输出约定，不能直接接新模型。
- 建议检测程序：加载 `best_int8.rknn` → 预处理 → 推理 → DFL+sigmoid+NMS（conf0.5/iou0.45）→ 画框输出，
  输入支持 图片/视频/`/dev/video0`，并可选开机自启（systemd）。
- **运行时版本**：板子 librknnrt 为 2.0.0b0，加载 2.3.2 模型有告警但功能正常。
