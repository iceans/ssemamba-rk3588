# 正确性检查

> **历史阶段说明（2026-09-24 补）**：本报告是旧协议阶段（方案 A）的正确性检查记录，属前期历史审计。方案 B 新协议（`SSE_DAUB_SERIES_B_20260915`，15 次独立完整训练）的预检实物以 `series_b/preflight/result.json` 及同目录逐模型预检 JSON（`B_FULL_0.json` / `B_3DCONV_0.json` / `E1.json` / `E2.json` / `E3.json`）为准；本报告不能替代新协议预检实物。新协议预检通过后，15 次训练与 15 个正式 TEST 均已 verify（见 `series_b/registry.json`）。

CPU：索引恢复和真实 v8DetectionLoss 的 7 组系数检查全部通过，包含一次反向传播。恒等 SSM 的 interleaved 与正确 blocked 恢复都得到 [5.5,11,16.5]；错误 even/odd 恢复得到 [15,15.5,2.5]。实际 IFSS/LFSS 当前求和而非平均，测试明确区分这一比例。非方形、多 batch/channel、转置、反向、merge 均已检查。未比较真实因果 SSM 在不同顺序下的数值相等性。

GPU：当前 LFSS n=1 与历史 LFSS n=4 的 kernel=1/3/5 均做了真实 SSM 前向、有限值和反向检查，只改补充分支卷积的 kernel/padding。历史 n=4 的块参数分别为 72576/76672/84864。STF 使用 4 次 DCN 对齐，3DConv 输出尺寸匹配。两份完整序列化历史模型的 640×640 前向成功；这不等于通过当前构造器或历史协议复现。

真实 DAUB 小样本：30 个首尾/跨序列边界位置，加载帧都在同一序列，关键帧标注吻合，坐标回原图误差小于 0.01 px。完整索引审计发现 4795 次请求只覆盖 4763 个不同关键帧，遗漏 32 帧，7 个关键帧重复。当前 Format 反转整个 8 通道张量，使时间分支顺序成为 i,i-1,...,i-4；STF 最后时间槽对应最早帧。不能悄悄修改后称为原协议。

E1 尚未通过完整变体检查：缺少可信的 LFSS×SAAM 原始定义，候选 SSTE checkpoint 虽无 CCCBlock，仍存有 Detect.aligned 模块，历史 forward 是否调用不能仅凭 state_dict 判断。

未进行任何正式训练。CPU 第一次 smoke 因测试脚本缺少 inchannel 参数失败，已修正并重跑；不计作模型训练失败。沙箱 CUDA 不可用的尝试同样不计入正式训练。全部检查日志保留在 evidence/。
