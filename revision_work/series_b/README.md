# 已批准的新 DAUB 实验系列

用户批准方案 B：15 次独立完整训练（原 13 次加两个缺失基准），100 epochs，种子 0/2027/3407；不使用已训练 DAUB 权重初始化。E1–E9 用 seed=0。两个额外组件基线、IRDST、新扫描顺序/位置消融和目标域训练不属于此次批准范围。

工作目录为 `/home/dell/lxs/tsgmamba/ultralytics-main`，解释器为 `/opt/anaconda3/envs/mambayolo/bin/python`。所有入口使用该已有环境，不安装/升级全局依赖。GPU 命令需要能访问 NVIDIA 设备的执行上下文。

```bash
/opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.prepare
CUDA_VISIBLE_DEVICES=1 /opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.preflight
/opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.prepare --freeze
/opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.queue --gpus 1 0
```

配置已生成且冻结后，不再次 prepare/freeze。队列进程持有文件锁，重复启动会拒绝；运行状态见 `queue.json` 和 `../STATUS.md`，每个任务的细粒度进度为 `runs/<ID>/progress.json`，stdout/stderr 位于该任务目录。队列检查 GPU 计算进程与空闲显存，不终止他人任务；低于 12 GiB 可用磁盘时暂停新增启动。

队列优先顺序：B_FULL_0 / B_3DCONV_0，R1/R2，R3/R4，然后 E1–E9。每个训练结束后自动运行对应 TEST；两个 seed=0 基准还会在共同 4683 个关键帧上执行 stride 1/2/4，并执行已冻结 NUDT-MIRSDT 官方测试集的 P3。全部训练和这些推理结束后，再在无其他计算进程时测量两模型的 P4 速度/显存。DFAR 来源缺口继续保留。

查看或单独恢复命令示例：

```bash
/opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.refresh
CUDA_VISIBLE_DEVICES=1 /opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.train --task B_FULL_0 --resume
CUDA_VISIBLE_DEVICES=1 /opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.evaluate --task B_FULL_0
```

不要在队列仍调度同一 GPU 时手工启动第二个任务。失败记录保留；队列不将失败任务标完成，也不自动更换 seed/超参数。已存在的兼容 `last.pt` 恢复模型、EMA、FP32 优化器、未累计完的梯度、随机状态和更新计数；每个 epoch 的 shuffle/翻转由 seed、epoch、关键帧索引确定。完成的正式权重为 `weights/selected_final.pt`，选择依据仅为固定 epoch=100 的 EMA。没有 test-selected best.pt。

新定义从已导出 frame5/3D 结构构造，LFSS n=4，IFSS 中未调用且历史不存在的 saam 成员不注册。DCN 和检测头 SAAM 显式调用，错误直接上报。E1 使用历史 SSTE YAML 的无 CCCBlock 路径，保留 Detect 的 SAAM。其余模块不按结果临时增删。

所有五帧按 `[g_i-4,g_i-3,g_i-2,g_i-1,g_i,R_i,G_i,B_i]` 排列。序列开头复制首帧，每个关键帧只出现一次；图像堆栈和关键帧框共同水平翻转。当前环境 Albumentations 因 albucore ImportError 未生效；当前项目也注释了 Mosaic/Perspective/HSV 的训练路径，新协议如实固定生效的水平翻转 0.5。

本系列使用独立的小型训练循环，调用原项目模型 parser、v8DetectionLoss 和 ModelEMA，并保留 Adam 分组、稳态有效 batch=64、warmup 累计和梯度裁剪。新 cosine 明确在第 100 epoch 到达 0.00001。这样能直接禁止训练中 test 选权，并完整保存恢复状态。不是调用当前会 resume JJB 的 `tsgmamba_train.py`。

`protocol.json` 的 source_files/source_sha256 冻结所有影响模型、数据、训练和指标计算的代码；队列/报表/检查脚本另存 workflow_files_at_freeze。`source_snapshot.tar.gz` 包含冻结时二者。每个正式任务启动及评估前核对源码、配置、模型 YAML 和数据清单摘要；不兼容即拒绝。原项目源码没有改动。

标准差仅在对应模型三个预定 seed 全部通过 TEST 后给出（样本标准差 ddof=1）。正式指标长表同时保留旧诊断来源，但仅 `status=verified, formal=True` 的新行能进入新协议论文统计。新结果段落自动生成于 `../writing/series_b_results_en.md`；审稿意见编号、稿件位置和未完成证据不自动宣称解决。

P3 数据与适配器单独冻结在 `external/protocol.json`：作者发布包及官方清单哈希、2,000 个测试关键帧、每张图像和 mask 的摘要、8 邻域 mask 转半开坐标框规则均已记录。使用源域预定阈值 0.25、相同的五帧输入和固定 seed=0 末轮权重；不训练目标域。`external/overlap_audit.json` 记录与 DAUB 共 13,777 张训练/测试图像的解码像素比较，无法排除背景素材经变换复用这一限制。

外部适配器为 `revision_work/new_protocol/external.py`，其哈希由外部协议独立约束；原训练、模型、数据和指标计算文件仍受最初 source_sha256 约束。更新后的调度器能通过 PID、进程命令和启动时间接管仍在运行的本系列任务，避免管理进程重启造成重复训练。此次接管记录为 `preflight/queue_reload_audit.json`，两个训练 PID 未变，训练重启次数为零。

独立执行外部测试的入口如下；在自动队列运行期间不要手工重复启动：

```bash
CUDA_VISIBLE_DEVICES=1 /opt/anaconda3/envs/mambayolo/bin/python -m revision_work.new_protocol.external --task B_FULL_0
```

外部测试结果保存至 `runs/<ID>/P3/`，通过核验后写入 `results/tables/external_generalization.csv` 及对应 LaTeX 表。正式外部推理须等待源域完整训练结束；数据冻结不代表已取得模型外部成绩。
