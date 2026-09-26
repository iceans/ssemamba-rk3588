# -*- coding: utf-8 -*-
"""
convgru_temporal.py — 消融 A6 专用: 与 TraceSS2D 同插槽的自由卷积递归对照
================================================================================
设计目标: 回答 "增益来自物理结构化的状态转移, 还是任意卷积递归都行?"
为保证公平, 本模块与 TraceSS2D 保持:
  * 相同的插槽与输入输出约定 ((B·T,C,H,W), keep='all'/'last');
  * 相同的帧内 SS2D 选项(打开时与 REF 的 P3 块逐参数一致, 差异只剩时序核);
  * 相同的残差结构与零初始化输出投影(插入即恒等, 训练起点一致);
  * 时序核参数量(扣除共享的 SS2D)实测: C=64 时 ConvGRU 59.8K vs TraceSS2D 33.5K,
    C=32 时 15.0K vs 9.1K —— 预算偏向 ConvGRU 一侧约 1.8×。这是对论证有利的
    公平方向: 给自由递归更多参数、它仍然更差, 结论才硬。groups=8 可把预算减半。

递归核为标准 ConvGRU:
  z = σ(W_hz*h + W_xz*x),  r = σ(W_hr*h + W_xr*x)
  h~ = tanh(W_hc*(r⊙h) + W_xc*x),   h ← (1−z)⊙h + z⊙h~

Ultralytics 注册(tasks.py):
  from ultralytics.nn.modules.convgru_temporal import ConvGRUTemporal
  parse_model 中追加:
        elif m is ConvGRUTemporal:
            c2 = ch[f]
            args = [ch[f], *args]
yaml 位置参数: [nframes, keep, use_ss2d, d_state, expand, groups]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .phys_ss2d import SS2D
except ImportError:
    from phys_ss2d import SS2D


class ConvGRUTemporal(nn.Module):
    def __init__(self, c1, nframes=5, keep="last", use_ss2d=False,
                 d_state=16, expand=1.0, groups=4):
        super().__init__()
        assert keep in ("all", "last")
        self.T, self.keep, self.use_ss2d = nframes, keep, use_ss2d

        if use_ss2d:                                    # 帧内空间: 与 TraceSS2D 一致
            self.ln = nn.LayerNorm(c1)
            self.ss2d = SS2D(c1, d_state=d_state, expand=expand)

        g = groups
        # 门控与候选: h 路与 x 路分开卷积再相加, 避免 cat 后分组把 h/x 切开
        self.h_zr = nn.Conv2d(c1, 2 * c1, 3, padding=1, groups=g, bias=False)
        self.x_zr = nn.Conv2d(c1, 2 * c1, 3, padding=1, groups=g)
        self.h_c = nn.Conv2d(c1, c1, 3, padding=1, groups=g, bias=False)
        self.x_c = nn.Conv2d(c1, c1, 3, padding=1, groups=g)

        self.norm = nn.GroupNorm(1, c1)
        self.out = nn.Conv2d(c1, c1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                   # 插入即恒等

    def _spatial(self, x):
        if not self.use_ss2d:
            return x
        return x + self.ss2d(self.ln(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

    def step(self, z_t, h=None):
        if h is None:
            h = torch.zeros_like(z_t)
        zr = self.h_zr(h) + self.x_zr(z_t)
        z, r = torch.sigmoid(zr).chunk(2, dim=1)
        h_cand = torch.tanh(self.h_c(r * h) + self.x_c(z_t))
        h = (1.0 - z) * h + z * h_cand
        return z_t + self.out(self.norm(h)), h

    def forward(self, x):                              # x: (B·T, C, H, W)
        # AMP(fp16) 训练时豁免 autocast、内部 fp32 运行(与 TraceSS2D 同策略,
        # 防止门控乘法/递归状态溢出污染 BN 统计量); 半精度评测模式直通。
        if torch.is_autocast_enabled():
            with torch.autocast("cuda", enabled=False):
                return self._forward(x.float()).to(x.dtype)
        return self._forward(x)

    def _forward(self, x):
        BT, C, H, W = x.shape
        T = self.T
        assert BT % T == 0, f"batch({BT}) 不能被 nframes({T}) 整除"
        zseq = self._spatial(x).view(BT // T, T, C, H, W)
        h, outs = None, []
        for t in range(T):
            o, h = self.step(zseq[:, t], h)
            outs.append(o)
        out = torch.stack(outs, 1)
        return out[:, -1] if self.keep == "last" else out.reshape(BT, C, H, W)


if __name__ == "__main__":
    import os
    os.environ["PHYS_SCAN_CHUNK"] = "16"
    torch.manual_seed(0)
    B, T, C, H, W = 2, 5, 32, 24, 24
    x = torch.randn(B * T, C, H, W, requires_grad=True)
    for keep, ss in (("all", False), ("last", True)):
        m = ConvGRUTemporal(C, nframes=T, keep=keep, use_ss2d=ss)
        y = m(x)
        y.mean().backward()
        print(f"keep={keep:4s} ss2d={ss}: {tuple(x.shape)} -> {tuple(y.shape)}")

    def core_params(m):                                 # 只数时序核(扣除 SS2D)
        skip = ("ss2d", "ln")
        return sum(p.numel() for n, p in m.named_parameters()
                   if not n.startswith(skip))

    try:
        from trace_modules import TraceSS2D
        for c in (32, 64):
            gru = ConvGRUTemporal(c, use_ss2d=False)
            trc = TraceSS2D(c, use_ss2d=False)
            print(f"C={c}: ConvGRU core {core_params(gru)/1e3:.1f}K  vs  "
                  f"TraceSS2D core {core_params(trc)/1e3:.1f}K")
    except ImportError:
        pass
