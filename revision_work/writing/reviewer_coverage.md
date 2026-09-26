# 审稿覆盖与证据状态（2026-09-24 更新）

仍缺 `codex/REVIEWER_ACCEPTANCE.md` 与投稿审稿原文（tex/PDF/bib），无法逐条还原 R1.1–R2.4 的确切 concern、编号与验收要求。本表不补造中间编号，也不把推测的关注点当作原评论。**没有任何实际审稿条目被标为已处理。** 三层状态说明见文末。

方案 B 新协议（`SSE_DAUB_SERIES_B_20260915`）已执行完毕：15 次独立完整训练、15 个正式 TEST 全部 verify，0 失败（`series_b/registry.json`）。

| 证据主题 | 当前实物 | 状态 |
|---|---|---|
| 资源与复用来源 | inventory、40 份重点权重审计、baseline_registry | 审计完成；正式 reusable=0 |
| 非交织恢复及替换边界 | implementation_audit、series_b/preflight/result.json | 索引/结构检查完成；历史 order-only 对照不合格或来源未明，保留为缺口 |
| 有效协议和公共种子 | series_b/protocol.json、series_b/approval.json | 已锁定；BASE_SEED=0，COMMON_SEEDS=[0,2027,3407] |
| 3 seed 重复（stf_repeats） | results/tables/stf_repeats.csv | 已完成：full 与 STF-3DConv 各 3 seed，均 verify |
| LFSS×SAAM 四组合（components） | E1（LFSS=0,SAAM=1）、B_FULL_0（1,1） | 缺 LFSS=0/SAAM=0 与 LFSS=1/SAAM=0 两行（补训未批准），四组合表不完整，保留为缺口 |
| 邻域 kernel（scan_neighborhood） | E2(k=1)、B_FULL_0(k=3)、E3(k=5) | 已完成；纯扫描顺序对照（order-only）仍缺，保留为缺口 |
| 损失敏感性（loss_sensitivity） | E4–E9 + B_FULL_0 默认行 | 已完成 6 组单因素，verify；默认权重未被替换 |
| P1 时序间隔 | full/3D 各 s∈{1,2,4}，共同 4683 帧 | 已完成 6 行；DFAR 三行 blocked（来源缺失） |
| P2 尺寸/场景分组 | grouping_lock、results/tables/size_scene_breakdown.csv | full/3D 已评估；d≤3 与 3<d≤7 为空组（0 框）；IFSS 非交织行与场景元数据仍缺 |
| P3 第三数据集外部测试 | external/protocol.json、P3 run/evaluation | full/3D 已冻结评估（NUDT-MIRSDT 官方测试 2000 帧）；DFAR 行 blocked；与 DAUB 背景复用无法完全排除 |
| P4 精度/成本 | P4/run.json（950 次原始计时） | full/3D 已完成；FLOPs 未可靠计数，保留局限 |
| 相关工作及主张收缩 | technical_comparison、manuscript_revision_snippets、reviewer_response_en | 草稿已形成；未改/编译原稿；未填实际审稿编号 |

仍缺的明确缺口：纯扫描顺序对照、融合位置（placement）消融、LFSS×SAAM 辅助模块组合补训（+2 次）、DFAR 比较基线、场景元数据、`REVIEWER_ACCEPTANCE.md` 与投稿源码。

三层状态：① 已生成运行实物（run.json、epochs.jsonl、TEST/evaluation.json、P1/P3/P4 记录、预检实物）为机器可核查记录；② 汇总一致（metrics.csv、results/tables、registry 状态）由产物自动汇总，字段一致；③ 独立复算/外部核验尚未完成——`formal=True` / `verified` 仅表示本协议内部状态机通过，不代表第三方已独立复算 AP。恢复验收文件后，每条实际评论需记录 exact concern、回应、修改内容、证据路径、稿件位置及限制。
