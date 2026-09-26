# TRACE 消融实验执行手册（按批次）

> 配套: `abl/` 下 15 个 yaml(由 `gen_ablations.py` 生成, 勿手改)、
> `run_ablations.py`(批次运行器)、三个模块文件(均已含 AMP 豁免补丁)。
> 原则: 每个变体只改 REF 的一处; 全部实验共用一套配方与验证协议。

---

## 0. 前置条件（一次性）

1. **模块**: `phys_ss2d.py`、`trace_modules.py`、`convgru_temporal.py` 放入
   `ultralytics/nn/modules/`（必须用本包内版本——含 half-eval 修复与 fp16
   autocast 豁免；旧版在 amp=True 下会复现 dtype 崩溃 / NaN 事故）。
2. **注册**: `tasks.py` 在既有 elif 基础上追加
   `from ...convgru_temporal import ConvGRUTemporal` 与
   `elif m is ConvGRUTemporal: c2 = ch[f]; args = [ch[f], *args]`。
3. **配方**: 一律使用 `run_ablations.py` 内的 RECIPE（AdamW lr0=0.002, cos_lr,
   150ep, patience=0, amp=True, mosaic=0, batch=16）。开跑后到日志里核对
   optimizer 行打印的是 `AdamW(lr=0.002...)` 而非 "found 'optimizer=auto'"。
4. **统一验证协议**: 所有论文数字从训练结束后的独立评测出——单卡、
   `val(batch=16, rect=False, conf/iou 默认)`，同一台机器同一脚本。
   训练内验证的数字只用于监控曲线。

---

## 1. 批次总览

| 批次 | 变体(seed) | 预计单卡串行时长 | 交付的论文素材 |
|---|---|---|---|
| 0 锚定系 | REF(0,1)、A7 早融(0,1)、A5 单帧(0) | ≈32 h | 主表坐标系 + 显著性标尺 |
| 1 脊柱 | A1 对角A(0)、A6 ConvGRU(0) | ≈17 h | "结构 vs 递归"核心表两行 |
| 2 算子分解 | A2 仅对流、A3 仅扩散、A4 各向同性 | ≈26 h | dissecting 表下半部 |
| 3 滤波组件 | B1 无参考系分解、B2 无新息门 | ≈17 h | 组件消融表 |
| 4 编码与头 | C1 普通stem、C2 仅SPD、D1 无P2头 | ≈20 h | 组件消融表 |
| 5 插入位置 | E1 仅P3、E2 仅P2 | ≈16 h | placement 表(与 REF 三行) |
| 6 暂缓 | 深监督/T扫描/背景差分开关/闪烁协议 | — | 见 §5 准备清单 |

时长按 3.5 min/epoch(TRACE 类)、1.5 min/epoch(A7)、3 min/epoch(A5/D1) 估算，
150 epoch，双卡 DDP 串行；批次 0–5 合计约 5.5 天。

**执行纪律**: 串行 DDP 双卡，batch 固定 16。不要为了并行把 batch 拆成 2×8 单卡
——per-GPU batch 变化会改变 BN 统计，重演 batch=4 事故；可比性优先于吞吐。

---

## 2. 各批次读数指南

### 批次 0（先跑，其他一切以它为参照）
三点定坐标: A5(无时序) → A7(通道堆叠) → REF(递归输运)。
- `REF − A5` = 时序信息总增益；`REF − A7` = 递归状态相对朴素堆叠的增益。
- REF 与 A7 各跑 seed∈{0,1}: 两 seed 差值就是后续所有单 seed 消融的
  **显著性标尺**——小于该差值的消融差异不写进正文结论。
- 若 REF(新配方) 相对 A7(新配方) 无优势: 先查曲线形态与 PR 曲线整条，再考虑
  DAUB 场景过于温和的可能（转向按尺寸/速度分桶与低 SNR 数据集），不要急着改结构。

### 批次 1（论文成立与否在此）
- A1 = 关 advection+diffusion（ego 随之关闭；**新息门保留**，与 REF 只差
  状态转移算子——论文文字须按此精确表述: "per-pixel diagonal transition with
  the same gated update"）。`REF − A1` = A 的空间结构增益。
- A6 = 同插槽/同残差/同 SS2D 选项的 ConvGRU，时序核参数量偏向 GRU 约 1.8×
  （C=64: 59.8K vs 33.5K，表中列出精确值）。给更多预算它仍更差，结论才硬。
- 读法: 若 REF>A6>A1 → 完整故事（结构有益，物理结构更优）；
  若 A6≈REF → 主张退守"物理约束在低数据/低 SNR/跨域泛化占优"，需批次 6 佐证。

### 批次 2
A2/A3 与 A1、REF 构成分解: 期望 A2(仅对流) 拿到大部分增益、A3(仅扩散) 小增益、
A4 说明各向异性的边际价值。若 A3>A2，重新审视 DAUB 的目标位移量级
（可能目标近似静止，扩散聚合就够——这本身是可写的分析）。

### 批次 3
- B1 建议**按序列分桶**评测（含明显相机运动的序列 vs 静止序列）——预期
  动桶显著、静桶无损，"安全性+必要性"一对证据。
- B2 常规 mAP 可能不敏感；完整故事等批次 6 的闪烁注入协议。

### 批次 4 / 5
C 组读 AP_tiny 端；D1 与 C 组解耦（头 vs 编码端）；E1/E2/REF 三行构成
"时序块放哪层"的 placement 表，预期 E1(仅 P3) 保留大部分增益、E2 说明
P2 层时序的边际贡献是否值它的显存。

---

## 3. 运行方式

```bash
python run_ablations.py 0      # 批次 0
python run_ablations.py 1      # 批次 1
python run_ablations.py all    # 顺序全跑
```
产物在 `runs/detect/abl_<TAG>_s<seed>/`。跑完每个批次后统一执行独立 val 并
登记 §4 表格。改配方 = 改 `run_ablations.py` 的 RECIPE 一处 + 重新
`python gen_ablations.py abl`（若涉及结构），已跑批次全部作废重跑——
这就是把配方收敛放在批次 0 之前完成的原因。

---

## 4. 结果登记表模板

| Tag | seed | mAP50 | mAP50-95 | P | R | Params(M) | ms/img | best ep | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| REF | 0 | | | | | | | | |
| REF | 1 | | | | | | | | |
| A7 | 0 | | | | | | | | |
| ... | | | | | | | | | |

论文表由此表按"REF 均值 ± 两 seed 半差"折算；效率列(ms/img)统一在同一张卡、
同一 batch 下测。

---

## 5. 批次 6 暂缓项的准备清单

1. **热图/速度深监督**: 按 `trace_modules.py` 文末伪代码接入自定义 Loss；
   数据侧需逐帧中心与相邻帧位移（由时序标注差分）。接入后以 REF+deep-sup
   为新变体（命名 REF-DS），预期是 mAP50-95 的下一个增量来源。
2. **T∈{3,8} 扫描**: 需重生成 labels 缓存(frame_num)、修改 gen_ablations.py
   顶部 `T=` 后重跑生成器（ch/nframes/TemporalSelect 会自动同步），
   产出 Pd–T 增益曲线（论文 Fig.4 素材）。
3. **背景差分开关(B3)**: 给 TraceSS2D 追加 `use_bgsub=True` 末位参数 +
   `if self.use_bgsub:` 包住 `Ut = Ut - self.bg_sub(Hb)`，属审稿追问储备。
4. **闪烁注入协议**: 测试序列随机帧加点噪声扫强度，报 Fa–强度曲线
   （B2 的招牌图）；只需评测脚本，无需重训。

---

## 6. 复查清单（每次开跑前 30 秒过一遍）

- [ ] 日志 optimizer 行 = AdamW（不是 "ignoring lr0..."）
- [ ] mosaic=0.0、batch=16、amp=True、patience=0、epochs=150
- [ ] 无 "未检测到 CUDA selective scan" 兜底警告
- [ ] 数据 yaml channels: 8 与 frame_num=5 一致
- [ ] name= 遵循 abl_<TAG>_s<seed>，不覆盖旧目录
- [ ] NaN 处置: 若再现 → 先确认模块为豁免版；仍现 → trainer 换 bf16
      (autocast dtype=torch.bfloat16 + GradScaler enabled=False)
