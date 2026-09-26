# -*- coding: utf-8 -*-
"""
run_ablations.py — 按批次串行执行消融实验
================================================================================
用法:
    python run_ablations.py 0        # 只跑批次 0(锚定系)
    python run_ablations.py 1        # 批次 1(论文脊柱)
    python run_ablations.py all      # 全部批次按序执行

前置(一次性, 详见 README_ablation.md):
    1) abl/*.yaml 已由 gen_ablations.py 生成;
    2) phys_ss2d.py / trace_modules.py / convgru_temporal.py(均为打过 AMP 豁免
       补丁的版本)已放入 ultralytics/nn/modules/, tasks.py 完成注册
       (含 ConvGRUTemporal 的 elif);
    3) 数据集 yaml 的 channels: 8 与 frame_num=5 已同步。

纪律: 所有实验共用 RECIPE, 只换 model 与 name/seed; batch 固定 16 不因并行拆小
(per-GPU batch 变化会改变 BN 统计, 破坏可比性); 串行 DDP 双卡执行。
论文数字统一从训练结束后的独立 val 协议出(见 README §统一验证协议)。
"""

import sys
from ultralytics import YOLO

RECIPE = dict(
    data="datacfg/DAUB.yaml", batch=16, imgsz=640, epochs=150,
    device=[0, 1], amp=True,
    optimizer="AdamW", lr0=0.002, lrf=0.01, cos_lr=True,
    warmup_epochs=5, patience=0, deterministic=False,
    mosaic=0.0, scale=0.2, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    frame_num=5,
    # cube=True,  # 你 fork 的自有参数按需补在这里, 保证所有实验一致
)

# (yaml, 标签, seed) —— 批次内按行顺序串行执行
BATCHES = {
    "0": [  # 锚定系: 三点定坐标(无时序/朴素时序/我们), REF 双 seed 给出显著性标尺
        ("abl/trace-ref.yaml",            "REF", 0),
        ("abl/trace-ref.yaml",            "REF", 1),
        ("abl/yolov8n-earlyfusion.yaml",  "A7",  0),
        ("abl/yolov8n-earlyfusion.yaml",  "A7",  1),
        ("abl/trace-a5-singleframe.yaml", "A5",  0),
    ],
    "1": [  # 论文脊柱: 状态转移算子的结构价值
        ("abl/trace-a1-diagA.yaml",   "A1", 0),
        ("abl/trace-a6-convgru.yaml", "A6", 0),
    ],
    "2": [  # 输运算子分解
        ("abl/trace-a2-advonly.yaml", "A2", 0),
        ("abl/trace-a3-difonly.yaml", "A3", 0),
        ("abl/trace-a4-isodiff.yaml", "A4", 0),
    ],
    "3": [  # 滤波器组件
        ("abl/trace-b1-noego.yaml",   "B1", 0),
        ("abl/trace-b2-noinnov.yaml", "B2", 0),
    ],
    "4": [  # 编码端与检测头
        ("abl/trace-c1-plainstem.yaml", "C1", 0),
        ("abl/trace-c2-spdonly.yaml",   "C2", 0),
        ("abl/trace-d1-nop2head.yaml",  "D1", 0),
    ],
    "5": [  # 插入位置
        ("abl/trace-e1-p3only.yaml", "E1", 0),
        ("abl/trace-e2-p2only.yaml", "E2", 0),
    ],
}


def run_one(yaml_path, tag, seed):
    name = f"abl_{tag}_s{seed}"
    print(f"\n{'=' * 68}\n===== {name}: {yaml_path}\n{'=' * 68}")
    YOLO(yaml_path).train(seed=seed, name=name, **RECIPE)


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "0"
    keys = list(BATCHES) if sel == "all" else [sel]
    failed = []
    for k in keys:
        for y, t, s in BATCHES[k]:
            try:
                run_one(y, t, s)
            except Exception as e:            # 单个失败不中断整批
                print(f"[FAIL] {t}_s{s}: {e}")
                failed.append(f"{t}_s{s}")
    if failed:
        print("\n失败清单(需人工排查后补跑):", failed)
