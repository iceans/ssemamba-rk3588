# 第三数据集接入与冻结方案

状态更新（2026-09-15）：已选择并下载 NUDT-MIRSDT 作者发布包，官方测试划分及适配器已冻结在 `../series_b/external/protocol.json`。完整模型与 STF-3DConv 的 P3 已加入队列，等待新 DAUB seed=0 的 100 epoch 训练和源域评价完成；DFAR 来源仍未核验。尚未运行目标模型成绩。以下候选比较保留接入决策记录。

| 官方候选 | 已核对信息 | 冻结前所缺 |
|---|---|---|
| NUDT-MIRSDT（已选） | 作者发布包包含逐帧 images、masks；官方清单为 80 个训练序列和 20 个测试序列。[作者仓库](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM) | 原始背景素材来源未逐一说明；无法排除经变换复用背景。实际测试清单已取得，不把 README 所称的 120 个总序列直接当作评价划分。 |
| IRSTD-UAV（未选） | TDCNet 作者仓库列出视频 images/sequence 与 labels/sequence 的 mask 目录约定。[作者仓库](https://github.com/IVPLabs/TDCNet) | 未下载，也未运行候选模型评分。 |

选择依据是官方清单、mask 标注和公开发布包可获取；没有依据模型成绩选择候选。已保存官方 URL、发布包 SHA256、train/test 清单 SHA256、图像及 mask 摘要。test.txt 的 `Sequence*/Mix/NNNNN.mat` 对应发布包 `images/NNNNN.png` 和 `masks/NNNNN.png`，20 个测试序列均含 1–100 号帧，共 2,000 个唯一关键帧，按官方 test 命名。

已冻结 mask→box 规则：原分辨率官方 mask 的非零像素；8 邻域连通域；每个连通域生成半开坐标 `[xmin,ymin,xmax+1,ymax+1]`，单像素为 1×1 非零框；类别为单类 target，GT 不设置信度。实际得到 1,729 个合法框：最大边 ≤3 为 101 个，(3,7] 为 977 个，>7 为 651 个。无目标帧保留在评价与误报分母中。

clip 沿用源域冻结的五帧时序，关键帧在末端，开头复制首帧，灰度关键帧复制为 RGB 通道，letterbox 到 640 后除以 255。完整模型、STF-3DConv 使用同一 DAUB 新协议 seed=0 固定第 100 epoch 的 EMA；DFAR 缺少来源证明，单独阻塞该行。目标域不训练、微调或选 epoch/seed/阈值。全帧误报工作点使用源域预先固定的 confidence=0.25、IoU=0.5；AP 解码及 NMS 与源域相同。

已对 2,000 张目标图像与 DAUB 训练/测试共 13,777 张图像比较包含尺寸及像素类型的解码图像哈希，未发现完全相同的图像。该检查无法识别所有裁剪、缩放或背景素材复用；作者 README 描述为合成数据，但没有逐一披露背景来源，因此不宣称已证明素材完全独立。证据位于 `../series_b/external/overlap_audit.json`。数据冻结和接入核验不等于已经完成外部模型评价。
