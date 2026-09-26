# 实现审计

> **历史阶段说明（2026-09-24 补）**：本报告完成于旧协议阶段（方案 A），是对历史 checkpoint 与当前源码的实现审计，属前期历史审计。方案 B 新协议（`SSE_DAUB_SERIES_B_20260915`）已按本审计结论（非交织/order-only 对照不合格、STF-3DConv 替换边界、E1 历史定义缺失等）建立独立统一协议并完成 15 次正式训练与 15 个 TEST。新协议实现约束见 `series_b/protocol.json`，新协议预检实物见 `series_b/preflight/result.json`。

本报告区分当前源码、checkpoint 中序列化的模块实例、历史训练行为三种证据。当前源码不等于历史训练源码；权重键存在不等于该层曾在历史 forward 中执行。原项目文件未修改，快照和逐文件 SHA256 见 `evidence/source_manifest.json`、`evidence/source_snapshot.tar.gz`。

## 1. STF→3DConv 的真实边界

从 checkpoint 导出的候选 YAML 位于 `evidence/checkpoints/`。逐层差异如下，层索引从零开始：

| 项目 | frame5 | conv3d-TIMSSM | fullsize_model |
|---|---|---|---|
| backbone[3] | TimeMambaStem，参数 [64, false] | Conv3dBlock，参数 [64,3,2,1] | TimeMambaStem，参数 [64] |
| head[4] Concat 来源 | [-1,9] | [-1,9] | [-1,10] |
| head[6] C2f YAML repeats | 6 | 6 | 3 |
| model.21.op.my_conv 层数 | 4 | 4 | 4 |

`frame5` 和 `conv3d-TIMSSM` 的架构 YAML 仅 backbone[3] 不同；`fullsize_model` 还改变 neck 路由和重复数。因此不能把后两者的 AP 或延迟差全归于 STF。`frame5` 只是结构匹配候选，未通过训练来源验收，不能据此自动替换论文的完整模型行。

[TimeMambaStem](../ultralytics/nn/modules/timeguid.py#L48) 依次包含四个空间 DWConv、以最后时间槽为参考的 DCN 对齐、时序 SSM 和输出投影。实际 3D 变体替换整个 stem，同时去除该处的 DWConv、DCN 和 SSM。[Conv3dBlock](../ultralytics/nn/modules/timeguid.py#L1669) 的实际路径为：对四维时间堆叠输入插入通道维，Conv3d，沿深度求均值，BatchNorm2d，SiLU。源码注释中的 BatchNorm3d/五维输入说明不符合实际 forward。

3D 变体仍保留 model.7、model.10 的 TMixSSM/IFSS，model.21 的 CCCBlock/LFSS，以及 Detect.aligned。它是两种 STF 设计的比较候选，不能称为全 CNN、完全移除 Mamba 或纯 SSM 算子对照。外围张量形状在 GPU 小测试中相同，输出均为 [2,16,16,24]，前向/反向有限；STF 的四次 DCN 调用成功。

## 2. 扫描与输出恢复

[SS2DguideMix](../ultralytics/nn/modules/timeguid.py#L1460) 对同形状特征按空间位置交织。两路为 `[m0,s0,m1,s1,...]`，多路为 `[x0[0],...,xT-1[0],x0[1],...]`；输出 reshape 到 `(B,C,L,T)` 后沿 T **求和**。

[SS2D1conv](../ultralytics/nn/modules/tgsmamba.py#L2108) 将主路与 `my_conv(x)` 的补充路交织，扫描长度 2L；输出 `ys[:,:,::2] + ys[:,:,1::2]`。当前构造器的补充 DWConv 默认 n=1；三个历史候选实例均为 n=4。E2/E3 必须保留锁定基准的 n，只变 kernel/padding。

`scripts/scan_pairing.py` 保存恒等映射下的正确 interleaved/blocked 参考实现。两路 `[10,20,30]`、`[1,2,3]` 各按自身索引恢复后平均均为 `[5.5,11,16.5]`。blocked 输入若仍按相邻 even/odd 平均，则得到 `[15,15.5,2.5]`。测试中平均仅用于验证索引，生产模型的 SUM 没有改成 MEAN。

当前 IFSS 中确有被注释的 blocked 拼接草稿；若启用草稿并保留当前配对会错。但缺乏对应历史源码，不能宣称已有 checkpoint 一定使用过这个错误。非方形、多 batch/channel、转置/反向路线恢复测试已通过。当前被调用的融合核心仅取 `cross_scan(... )[:,0]`，k_group=1，不能因公共 helper 支持四方向就把实际模型描述为四方向或双向。

[SS2D1conv_wosfi](../ultralytics/nn/modules/tgsmamba.py#L2436) 实际注释了补充分支调用，仅保留主路、长度 L，亦不再做两路配对。因此**当前定义不满足纯扫描顺序对照**。恒等核心检查实际观测正常 LFSS 长度 30、wosfi 长度 15。相关旧权重不能用修复后的 forward 重评后冒称有效顺序消融；需恢复可追溯的原始定义，或明确新增独立训练。

## 3. 完整加载与历史训练

40 份重点权重中，9 份可由当前 parser 严格重建，31 份失败；这只是张量兼容检查，未证明历史协议有效。`fullsize_model` 和 3D 候选的当前构造器缺少/多出参数：TMixSSM 新增但 forward 未使用的 `saam`，以及 LFSS n=1 对历史 n=4 的差异。两次诊断采用完整序列化模型实例恢复，严格比对所有 state_dict 键、形状和值；没有使用 `strict=False`。序列化模型仍调用当前类方法，因此也不是历史代码快照的替代物。

fullsize/3D/frame5 候选训练配置为 20 epochs、seed=0、batch=8、optimizer=auto、lr0=0.0001、cos_lr=False，初始化指向已经在 DAUB 训练的完整模型 checkpoint。它们不能充当独立完整消融或有效三种子统计中的一个种子。`my_2timMix_945` 虽配置 100 epochs，但 batch=32、跨数据集初始化、模块定义不同，尚不能认定为相同完整模型。剥离后的 checkpoint optimizer 为 None，不能依据今天的 auto 规则补写历史实际优化器。

`wodwconv`、`wodwconv_w_dcn` 实例均仍有四个 spatial_dw 权重而没有 alignment；`dcn` 同时有两者。这些目录名无法直接映射 w/o DWConv 和 w/o DCN。EMA 的 BN 计数器不能用来判定模块是否执行。

## 4. 加载器与评价

DAUB YAML 的 val 指向 `test/images`，不存在独立 val 或 test 键。4795 次验证请求实际覆盖 4763 个不同关键帧，遗漏 32 帧、7 个关键帧有重复。两次缓存均保留每次请求和空预测帧；不能去重后直接沿用已计算 AP。

[Format._format_img](../ultralytics/data/augment.py#L1686) 反转整个 8 通道张量。最终网络输入为 `[gray_i,gray_i-1,...,gray_i-4,key_R,key_G,key_B]`，STF 的最后时间槽实际为最早帧。已检查的边界样本都来自同一序列、关键帧标签吻合、坐标回原图误差 <0.01 px；这不消除完整样本集合重复/遗漏。clip 几何增强的一致性仍需对最终锁定训练加载器核验。

统一诊断重评固定 conf=0.001、NMS IoU=0.7、640、T=5、FP32、batch=8。3D AP50=94.868%，接近权重内嵌验证值 94.876%；fullsize AP50=87.588%，低于内嵌 94.998% 约 7.41087 个百分点。差异原因尚未完整定位；没有扫描阈值、挑片段或换 seed 追分。上述结果不具有独立测试资格。

## 5. 其他会改变实验含义的实现细节

`TimeMambaStem.forward` 的 DCN 和 [Detect.forward](../ultralytics/nn/modules/head.py#L339) 的 SAAM 均使用宽泛 `except: pass`，可能静默跳过模块。新协议应显式记录调用或失败；不能单靠权重键宣称历史辅助模块执行过。诊断重评保留了当前路径，没有偷偷删改模块恢复指标。

[SelectiveScanCuda](../ultralytics/nn/modules/tgsmamba.py#L1096) 实际固定调用 `selective_scan_cuda_oflex` 并转 FP32，忽略 backend 选择参数；环境记录据此填写。CPU 理论参考 `selective_scan_torch` 使用 `exp(delta*A)` 和 `delta*B*u` 的递推，论文片段与此一致。

实际损失检查证明 E4–E9 的 box/cls/dfl 系数到达 v8DetectionLoss，各指定分量仅变为 0.5 倍或 2 倍，其他分量不变，且反向有效。E1 仍缺可信的 LFSS 基线块与 SAAM 开关定义，未把候选目录拼接当作已通过的 E1。

统一成本结果见 `results/tables/candidate_runtime.csv` 和 `benchmark_protocol.json`。只有候选模型图的独立测量意义；完整模型与 3D 的图差异、未审计 SSM/DCN FLOPs 和缓存分配器复用均已说明，不引用旧 FPS 混表。
