# 已执行与恢复命令

工作目录：`/home/dell/lxs/tsgmamba/ultralytics-main`；解释器：`/opt/anaconda3/envs/mambayolo/bin/python`。

以下都是实际入口，无未解析占位符。审计脚本会重新生成自身报告；诊断重评目录若已存在会拒绝覆盖，重启时先读取 run_registry.json 和 runs/*/run.json，并用 `nvidia-smi` / `ps` 检查进程。

```bash
python revision_work/scripts/audit_checkpoints.py
python revision_work/scripts/audit_data.py
python revision_work/scripts/audit_samples.py
python revision_work/scripts/preflight.py --device cpu
python revision_work/scripts/preflight.py --device cuda:1
python revision_work/scripts/diagnostic_reeval.py --model fullsize_model --device 1
python revision_work/scripts/diagnostic_reeval.py --model conv3d-TIMSSM --device 1
python revision_work/scripts/build_deliverables.py
python revision_work/scripts/finalize_artifacts.py
```

GPU 命令必须处于可以访问 NVIDIA 设备的执行上下文，已使用沙箱外权限运行。所有新文件均写入 revision_work。dataset 原目录没有被修改，必要的新缓存重定向到 evidence/。

E1–E9、R1–R4 尚无可诚信给出的正式启动命令：有效基准、准确网络定义、训练与选权协议未锁定，而且缺少语义清单。resolved_experiments.json 对 command 使用 null 并给出原因，不能将其误当 ready。当前 tsgmamba_train.py 会恢复 JJB 训练，禁止用它代替本任务入口。
