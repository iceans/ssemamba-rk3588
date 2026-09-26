# 资源清单

记录时间：2026-09-14T19:23:32.399009+00:00。仓库目录：`/home/dell/lxs/tsgmamba/ultralytics-main`。

未发现当前目录及祖先目录或项目相关发布副本中的 AGENTS.md。`.git/` 在当前环境为空，`git status` 失败，因此无法核验 commit/未提交改动或创建分支；没有执行 init/reset/clean。保留全部原文件，当前源文件摘要为 `2b9984b1e1dea3dacae3159c1083a01ffa25a93455ef0ed9a546e83512e816b5`，备份见 `evidence/source_snapshot.tar.gz`。

真实训练入口是 `tsgmamba_train.py`，目前激活的是 JJB、7 帧、resume=True 的任务，不能直接运行用于此次实验。模型配置为 Ultralytics YAML + `YOLO.train(**kwargs)`。当前验证入口是 `YOLO.val` / `ultralytics/models/yolo/detect/val.py`。`fpstest.py` 可查到旧速度口径，但其注释与输入参数存在不一致。

Python `/opt/anaconda3/envs/mambayolo/bin/python`，3.11.11；PyTorch 2.0.0+cu118，torchvision 0.15.1+cu118，CUDA build 11.8。两张 RTX 4090，每张约 24 GiB，驱动 535.288.01；核查时无计算任务，GPU 0 有桌面进程。GPU 1 已执行真实 SSM/DCN 前向反向与推理。CUDA 在沙箱内不可用，在已获准的沙箱外运行可用。未升级或安装全局依赖。完整环境见 `evidence/environment.json` 和 `pip_freeze.txt`。工作盘剩余约 97.4 GiB。

发现 720 份 args.yaml、928 个 pt/pth 文件（含重复副本，不能视为独立有效实验）；对 40 个重点 checkpoint 做了哈希、参数图、训练配置、严格构造检查，其中 9 个通过当前构造器的张量严格匹配，但没有一个已通过完整科学复用验收。详见 baseline_registry.json。

DAUB：`/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format`，train 8982 帧/10 序列，test 4795 帧/7 序列；YOLO 归一化框。当前 val 指向 test，没有独立 val。IRDST：`/home/dell/lxs/Anti_UAV_dataset/IRDST/yolo_dataset_png/images`，train 20398 帧/42 序列，val 20258 帧/43 序列；labels 在同级 labels 目录。没有配置 test。划分序列 ID 无交集；尚不能排除改名或派生背景重叠。逐帧标注哈希、原尺寸、框、序列和 P1 共同集合见 evidence/data/。根目录同名 DAUB 和 IRDST 实际是 PNG 图，不是数据目录。

DFAR：`/home/dell/lxs/Anti_UAV_compare_method/DFAR-main`，DAUB 候选权重 `logs_updated/DAUB/ep010-loss1.974-val_loss1.744.pth` 的 628 项 state_dict 严格加载通过；对应训练 provenance、论文使用的是哪个 checkpoint 尚未锁定，脚本 seed=2023 不能当作该 checkpoint 的有效历史 seed。

相关发布副本：`/home/dell/lxs/tsgmamba/tsgmamba-release`，提供 README/train/val，但不是可追溯的投稿版本。UAVSwarm、STtran、ITSDT 视频目录存在；尚未认定为适用于本任务且独立的第三红外视频测试协议。指定相关目录未发现 NUDT-MIRSDT/IRSTD-UAV 已适配数据。

缺失：`codex/experiment_manifest.json`、`codex/REVIEWER_ACCEPTANCE.md`、投稿论文 PDF/tex/bib、与有效原始训练对应的不可变源代码/配置和划分摘要。论文图片和 PR 数组存在，Ours.npy 只有 precision/recall，不能追溯 checkpoint 或逐帧预测来源。没有向无关目录或账户搜索。
