# -*- coding: utf-8 -*-
"""
gen_ablations.py — 从参考配置一键生成全部消融 yaml
================================================================================
原则: 每个变体只改参考配置(REF)的一处; 所有文件由本脚本生成, 避免手改 15 份
yaml 时的索引/参数错位。修改配方后重跑本脚本即可同步全部变体。

用法:  python gen_ablations.py [输出目录, 默认 ./abl]
生成后自动做两级校验: yaml 可解析 + Detect 引用层存在。
"""

import sys
import yaml
from pathlib import Path

T = 5            # 帧数(与数据集 frame_num、ch=T+3 同步)
CH = T + 3

# ---------------------------------------------------------------- 层定义积木
STEM_HF  = ["  - [-1, 1, HFStem, [64]]                            # 1  P1/2 M1: DoG+SPD",
            "  - [-1, 1, SPDConv, [128]]                          # 2  P2/4 M1"]
STEM_SPD = ["  - [-1, 1, SPDConv, [64]]                           # 1  P1/2 仅SPD(无DoG)",
            "  - [-1, 1, SPDConv, [128]]                          # 2  P2/4"]
STEM_CONV= ["  - [-1, 1, Conv, [64, 3, 2]]                        # 1  P1/2 普通stem",
            "  - [-1, 1, Conv, [128, 3, 2]]                       # 2  P2/4"]

# TraceSS2D args: [nframes,d_state,expand,groups,v_max,keep,substeps,bg_ratio,
#                  use_ss2d,use_ego,use_innov,aniso,use_advection,use_diffusion]
def trace(dstate, keep, ss2d, ego, innov, aniso, adv=True, dif=True):
    a = [T, dstate, 1.0, 4, 3.0, f"'{keep}'", 2, 0.25,
         ss2d, ego, innov, aniso]
    if not (adv and dif):
        a += [adv, dif]
    return "[" + ", ".join(str(x) for x in a) + "]"

L4_REF = f"  - [-1, 1, TraceSS2D, {trace(8,  'all',  False, False, True,  False)}]"
L7_REF = f"  - [-1, 1, TraceSS2D, {trace(16, 'last', True,  True,  True,  True )}]"

HEAD_P2 = f"""head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 13
  - [[-1, 9], 1, Concat, [1]]                         # 14
  - [-1, 3, C2f, [512]]                               # 15
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 16
  - [[-1, 7], 1, Concat, [1]]                         # 17
  - [-1, 3, C2f, [256]]                               # 18 P3 中间
  - [4, 1, TemporalSelect, [{T}]]                     # 19 P2 支路取末帧
  - [18, 1, nn.Upsample, [None, 2, "nearest"]]        # 20
  - [[-1, 19], 1, Concat, [1]]                        # 21
  - [-1, 3, C2f, [128]]                               # 22 P2/4 输出
  - [-1, 1, Conv, [128, 3, 2]]                        # 23
  - [[-1, 18], 1, Concat, [1]]                        # 24
  - [-1, 3, C2f, [256]]                               # 25 P3/8 输出
  - [-1, 1, Conv, [256, 3, 2]]                        # 26
  - [[-1, 15], 1, Concat, [1]]                        # 27
  - [-1, 3, C2f, [512]]                               # 28 P4/16 输出
  - [-1, 1, Conv, [512, 3, 2]]                        # 29
  - [[-1, 12], 1, Concat, [1]]                        # 30
  - [-1, 3, C2f, [1024]]                              # 31 P5/32 输出
  - [[22, 25, 28, 31], 1, Detect, [nc]]               # 32 P2-P5"""

HEAD_P3 = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 13
  - [[-1, 9], 1, Concat, [1]]                         # 14
  - [-1, 3, C2f, [512]]                               # 15
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 16
  - [[-1, 7], 1, Concat, [1]]                         # 17
  - [-1, 3, C2f, [256]]                               # 18 P3/8 输出
  - [-1, 1, Conv, [256, 3, 2]]                        # 19
  - [[-1, 15], 1, Concat, [1]]                        # 20
  - [-1, 3, C2f, [512]]                               # 21 P4/16 输出
  - [-1, 1, Conv, [512, 3, 2]]                        # 22
  - [[-1, 12], 1, Concat, [1]]                        # 23
  - [-1, 3, C2f, [1024]]                              # 24 P5/32 输出
  - [[18, 21, 24], 1, Detect, [nc]]                   # 25 P3-P5(无P2头)"""

TEMPLATE = """# =============================================================================
# {fname} —— 消融变体 {tag}
# 改动: {change}
# 理由: {why}
# 由 gen_ablations.py 生成, 请勿手改; 配方变更后重跑生成器。
# =============================================================================
nc: 1
ch: {ch}        # = nframes({t}) + key_channels(3)
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

backbone:
  - [-1, 1, TemporalUnfold, [{t}, 3, 3]]              # 0  (B,{ch},H,W)->(B*{t},3,H,W)
{stem1}
{stem2}
  - [-1, 3, C2f, [128, True]]                         # 3
{l4}
  - [-1, 1, Conv, [256, 3, 2]]                        # 5  P3/8
  - [-1, 6, C2f, [256, True]]                         # 6
{l7}
  - [-1, 1, Conv, [512, 3, 2]]                        # 8  P4/16
  - [-1, 6, C2f, [512, True]]                         # 9
  - [-1, 1, Conv, [1024, 3, 2]]                       # 10 P5/32
  - [-1, 3, C2f, [1024, True]]                        # 11
  - [-1, 1, SPPF, [1024, 5]]                          # 12

{head}
"""

IDT  = "  - [-1, 1, nn.Identity, []]                          # 4  (占位, 保持索引)"
SEL7 = f"  - [-1, 1, TemporalSelect, [{T}]]                    # 7  取末帧(无时序建模)"
GRU4 = f"  - [-1, 1, ConvGRUTemporal, [{T}, 'all', False]]     # 4  自由卷积递归对照"
GRU7 = f"  - [-1, 1, ConvGRUTemporal, [{T}, 'last', True]]     # 7  自由卷积递归对照"

# ------------------------------------------------------- 变体表: 单变量原则
V = {}
def add(tag, fname, change, why, l4=L4_REF, l7=L7_REF,
        stem=STEM_HF, head=HEAD_P2):
    V[tag] = dict(fname=fname, change=change, why=why,
                  l4=l4, l7=l7, stem=stem, head=head)

add("REF", "trace-ref",
    "无(参考配置: P2块无SS2D+新息门, P3块全配置)",
    "所有对比的锚点")
add("A1", "trace-a1-diagA",
    "两块关 advection+diffusion(ego随之关闭)",
    "核心命题: 增益来自A的空间结构还是任意递归; 与REF只差状态转移算子",
    l4=f"  - [-1, 1, TraceSS2D, {trace(8,'all',False,False,True,False,False,False)}]",
    l7=f"  - [-1, 1, TraceSS2D, {trace(16,'last',True,False,True,False,False,False)}]")
add("A5", "trace-a5-singleframe",
    "P2块->Identity, P3块->TemporalSelect",
    "单帧锚点: stem/头完全相同, 量化时序信息总增益",
    l4=IDT, l7=SEL7)
add("A6", "trace-a6-convgru",
    "两个TraceSS2D->ConvGRUTemporal(时序核参数量偏向GRU约1.8x)",
    "物理约束 vs 自由卷积递归; 回应'不就是花哨的ConvGRU'",
    l4=GRU4, l7=GRU7)
add("A2", "trace-a2-advonly",
    "两块关 diffusion(aniso随之关闭)",
    "输运项单独贡献(沿轨迹搬运能量)",
    l4=f"  - [-1, 1, TraceSS2D, {trace(8,'all',False,False,True,False,True,False)}]",
    l7=f"  - [-1, 1, TraceSS2D, {trace(16,'last',True,True,True,False,True,False)}]")
add("A3", "trace-a3-difonly",
    "两块关 advection(ego/aniso随之关闭)",
    "扩散项单独贡献; 亦即vHeat式纯扩散+时间递归的对照",
    l4=f"  - [-1, 1, TraceSS2D, {trace(8,'all',False,False,True,False,False,True)}]",
    l7=f"  - [-1, 1, TraceSS2D, {trace(16,'last',True,False,True,False,False,True)}]")
add("A4", "trace-a4-isodiff",
    "仅P3 aniso=False",
    "各向异性(沿运动方向的管状扩散)是否优于各向同性",
    l7=f"  - [-1, 1, TraceSS2D, {trace(16,'last',True,True,True,False)}]")
add("B1", "trace-b1-noego",
    "仅P3 use_ego=False",
    "M2参考系分解的价值; 建议按'含相机运动的序列'分桶读数",
    l7=f"  - [-1, 1, TraceSS2D, {trace(16,'last',True,False,True,True)}]")
add("B2", "trace-b2-noinnov",
    "两块 use_innov=False(退回普通sigmoid门)",
    "M4新息门的价值; 常规mAP或不敏感, 后续配闪烁注入协议",
    l4=f"  - [-1, 1, TraceSS2D, {trace(8,'all',False,False,False,False)}]",
    l7=f"  - [-1, 1, TraceSS2D, {trace(16,'last',True,True,False,True)}]")
add("C1", "trace-c1-plainstem",
    "HFStem/SPDConv -> 两个stride-2 Conv",
    "M1整体(无损下采样+DoG先验)对小目标端的贡献",
    stem=STEM_CONV)
add("C2", "trace-c2-spdonly",
    "第1层 HFStem -> SPDConv(保留无损下采样, 去掉DoG)",
    "从M1中单独剥出DoG对比先验的贡献",
    stem=STEM_SPD)
add("D1", "trace-d1-nop2head",
    "检测头改回P3-P5三尺度(骨干含P2时序块不动)",
    "P2头对tiny目标的贡献, 与C组(编码端)解耦",
    head=HEAD_P3)
add("E1", "trace-e1-p3only",
    "P2块->Identity(仅P3做时序)",
    "插入位置表: 高分辨率层的时序建模是否必要",
    l4=IDT)
add("E2", "trace-e2-p2only",
    "P3块->TemporalSelect(仅P2做时序)",
    "插入位置表: 反方向",
    l7=SEL7)

EARLYFUSION = """# =============================================================================
# yolov8n-earlyfusion.yaml —— A7: 8通道早期融合基线(train7的可复现副本)
# 理由: 递归状态 vs 通道堆叠; 旧train7数字属旧配方, 必须在统一配方下重跑。
# =============================================================================
nc: 1
ch: 8
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]
  - [[15, 18, 21], 1, Detect, [nc]]
"""


def main(outdir="abl"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for tag, v in V.items():
        text = TEMPLATE.format(
            fname=v["fname"] + ".yaml", tag=tag, change=v["change"],
            why=v["why"], ch=CH, t=T, stem1=v["stem"][0], stem2=v["stem"][1],
            l4=v["l4"], l7=v["l7"], head=v["head"])
        p = out / (v["fname"] + ".yaml")
        p.write_text(text, encoding="utf-8")
        d = yaml.safe_load(text)                          # 校验1: 可解析
        n_layer = len(d["backbone"]) + len(d["head"])
        det = d["head"][-1]
        assert det[2] == "Detect" and all(i < n_layer - 1 for i in det[0]), \
            f"{tag}: Detect 引用越界"                      # 校验2: 引用存在
        ok += 1
        print(f"[ok] {tag:4s} -> {p.name}  ({n_layer} layers)")
    (out / "yolov8n-earlyfusion.yaml").write_text(EARLYFUSION, encoding="utf-8")
    yaml.safe_load(EARLYFUSION)
    print(f"[ok] A7   -> yolov8n-earlyfusion.yaml")
    print(f"共生成 {ok + 1} 个配置, 输出目录: {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "abl")
