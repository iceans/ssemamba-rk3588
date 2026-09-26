# -*- coding: utf-8 -*-
"""
trace_v2.py — TRACE-v2: 证据驱动重设计的选择性输运时序块
================================================================================
设计依据(DAUB 统一口径消融, 2026-07):
  * B2(无新息门) 0.9279/0.6145 全场最优, REF 0.9030/0.6033
      → 删除新息门(M4)。机制: 新目标入场无历史 → 新息大 → 门关闭,
        系统性压制目标出现的头几帧(经典滤波的"航迹起始"问题被硬编码成了伤害)。
        更新门回归 Mamba 式输入依赖选择性: g = sigmoid(coef(z_t))。
  * A4(0.6029) > A1(0.5945) > A2(0.5856): 对流与扩散是耦合对——
      速度不完美时 warp 搬离真值、纯放大误差, 扩散兜住速度误差后才净增益
      → 对流+扩散作为整体保留, 不再拆分开关。
  * A4 ≈ REF(0.6029 vs 0.6033): 各向异性无证据 → 删除, 换取简洁与速度。
  * B1(0.5880) < REF: 伽利略分解有效 → 保留 ego 头。
  * E1(仅P3, 0.6123) > REF(P2+P3, 0.6033): P2 时序块有害 → 本块只部署在 P3。
  * A6 ConvGRU ≈ REF: 物理结构不买"更准", 买的是零参数时序核 + 非扩张稳定性
      + 零样本 rollout 预测 + 可解释速度场 → rollout 能力完整保留。

设计原则一句话: **物理管搬运(转移算子), 学习管写入(更新门)。**

保留的数值安全措施(生产事故沉淀):
  * AMP 训练时整块豁免 autocast 内部 fp32(fp16 门控递归溢出 → BN 污染 → NaN);
  * ego 头输入跟随权重 dtype(验证期整模型 .half() 兼容);
  * _base_grid 真实 device 校验 + __getstate__ 清缓存(跨设备 save/load);
  * v=tanh封顶 / α=0.25·sigmoid(CFL) / λ=softplus / g=sigmoid → 非扩张稳定。

Ultralytics 注册(tasks.py):
    from ultralytics.nn.modules.trace_v2 import TraceV2
    parse_model 追加:
        elif m is TraceV2:
            c2 = ch[f]
            args = [ch[f], *args]
yaml 位置参数: [nframes, d_state, expand, groups, v_max, keep,
               diff_substeps, bg_ratio, use_ss2d, use_ego, heat_aux]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .phys_ss2d import SS2D
except ImportError:
    from phys_ss2d import SS2D


class TraceV2(nn.Module):
    """选择性输运时序块 v2: 半拉格朗日对流 + 各向同性显式扩散 + 衰减
    + 输入依赖门控注入; 背景组慢 EMA + 可微背景差分; 可选帧内 SS2D 与热图读出。
    输入 (B·T, C, H, W), keep='last' 输出 (B, C, H, W) / 'all' 输出同形。"""

    def __init__(self, c1, nframes=5, d_state=16, expand=1.0, groups=4,
                 v_max=3.0, keep="last", diff_substeps=2, bg_ratio=0.25,
                 use_ss2d=True, use_ego=True, heat_aux=False):
        super().__init__()
        assert keep in ("all", "last") and c1 % groups == 0
        self.T, self.G, self.n = nframes, groups, c1
        self.v_max, self.k_diff, self.a_cfl = v_max, diff_substeps, 0.25
        self.keep, self.use_ss2d, self.use_ego = keep, use_ss2d, use_ego
        self.heat_aux = heat_aux
        cpg = c1 // groups
        self.g_bg = max(1, round(bg_ratio * groups))
        self.g_tg = groups - self.g_bg
        assert self.g_tg >= 1, "bg_ratio 过大, 目标组至少保留 1 组"
        self.n_bg, self.n_tg = self.g_bg * cpg, self.g_tg * cpg

        # ---- 帧内空间编码(可选, 与 v1 逐参数一致) ----
        if use_ss2d:
            self.ln = nn.LayerNorm(c1)
            self.ss2d = SS2D(c1, d_state=d_state, expand=expand)

        # ---- 系数头: v_l(2·g_tg) + α_bg(g_bg) + α_tg(g_tg) + λ(C) + g(n_tg) ----
        self.n_vl = 2 * self.g_tg
        n_coef = self.n_vl + self.g_bg + self.g_tg + c1 + self.n_tg
        self.coef = nn.Conv2d(c1, n_coef, 1)
        nn.init.zeros_(self.coef.weight)          # 零初始化: v=0, α=0.125, 门=0.5
        nn.init.zeros_(self.coef.bias)            # 起步 ≈ 温和的逐点 EMA

        if use_ego:                                # M2: 全局 6 参仿射(相机自运动)
            self.ego = nn.Sequential(nn.Linear(c1, 32), nn.SiLU(),
                                     nn.Linear(32, 6))
            nn.init.zeros_(self.ego[-1].weight)
            nn.init.zeros_(self.ego[-1].bias)

        self.U = nn.Conv2d(c1, c1, 1)              # 观测注入
        self.bg_sub = nn.Conv2d(self.n_bg, self.n_tg, 1, bias=False)
        nn.init.zeros_(self.bg_sub.weight)         # 可微背景差分, 零初始化

        # 固定 5 点拉普拉斯模板(分组逐通道)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("lap_bg", lap.expand(self.n_bg, 1, 3, 3).clone())
        self.register_buffer("lap_tg", lap.expand(self.n_tg, 1, 3, 3).clone())

        self.norm = nn.GroupNorm(1, c1)
        self.out = nn.Conv2d(c1, c1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)              # 插入即恒等

        if heat_aux:                               # M5 读出(批次7 档二依赖)
            self.heat = nn.Conv2d(self.n_tg, 1, 1)
        self._frames_aux, self.last_v = None, None
        self._grid_cache = {}

    # ---------------- 网格与搬运 ----------------
    def _base_grid(self, h, w, dev):
        key = (h, w, str(dev))
        g = self._grid_cache.get(key)
        if g is None or g.device != dev:
            ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=dev),
                                    torch.arange(w, dtype=torch.float32, device=dev),
                                    indexing="ij")
            g = torch.stack((xs, ys), -1)          # (h,w,2) [x,y]
            self._grid_cache[key] = g
        return g

    def __getstate__(self):                        # 缓存不随 .to() 迁移, 序列化前清空
        st = self.__dict__.copy()
        st["_grid_cache"] = {}
        return st

    def _advect_by(self, field, v, groups):
        """半拉格朗日搬运任意分组场; 双线性采样=凸组合, 不放大界。"""
        B, N, h, w = field.shape
        cpg = N // groups
        Hg = field.float().reshape(B * groups, cpg, h, w)
        vg = v.float().reshape(B * groups, 2, h, w).permute(0, 2, 3, 1)
        src = self._base_grid(h, w, field.device) - vg
        gx = src[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
        gy = src[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        out = F.grid_sample(Hg, torch.stack((gx, gy), -1), mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.view(B, N, h, w).to(field.dtype)

    # ---------------- 系数与输运 ----------------
    def _coefs(self, z_t):
        B, C, h, w = z_t.shape
        coef = self.coef(z_t)
        o = 0
        vl = self.v_max * torch.tanh(coef[:, o:o + self.n_vl]); o += self.n_vl
        a_bg = self.a_cfl * torch.sigmoid(coef[:, o:o + self.g_bg]); o += self.g_bg
        a_tg = self.a_cfl * torch.sigmoid(coef[:, o:o + self.g_tg]); o += self.g_tg
        lam = F.softplus(coef[:, o:o + C]); o += C
        gate = torch.sigmoid(coef[:, o:])                            # (B,n_tg,h,w)

        base = self._base_grid(h, w, z_t.device)
        xn = base[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
        yn = base[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        if self.use_ego:
            # 输入跟随权重 dtype(兼容验证期 .half()), 输出 fp32 与网格坐标运算
            w_dtype = self.ego[0].weight.dtype
            p = self.ego(z_t.mean((2, 3)).to(w_dtype)).float()       # (B,6)
            vgx = p[:, 0, None, None] * xn + p[:, 1, None, None] * yn + p[:, 2, None, None]
            vgy = p[:, 3, None, None] * xn + p[:, 4, None, None] * yn + p[:, 5, None, None]
            v_g = self.v_max * torch.tanh(torch.stack([vgx, vgy], 1)).to(z_t.dtype)
        else:
            v_g = z_t.new_zeros(B, 2, h, w)
        v_tg = v_g.unsqueeze(1) + vl.view(B, self.g_tg, 2, h, w)
        v_bg = v_g.unsqueeze(1).expand(B, self.g_bg, 2, h, w)
        v_all = torch.cat([v_bg, v_tg], 1).reshape(B, 2 * self.G, h, w)
        return dict(vl=vl, a_bg=a_bg, a_tg=a_tg, lam=lam, gate=gate,
                    v_g=v_g, v_tg=v_tg, v_all=v_all)

    def _propagate(self, state, co):
        """预测步 = 对流 + 各向同性扩散(耦合对, 依据 A2/A4 不再拆分)。"""
        cpg = state.shape[1] // self.G
        H = self._advect_by(state, co["v_all"], self.G)
        Hb, Ht = H[:, :self.n_bg], H[:, self.n_bg:]
        ab = co["a_bg"].repeat_interleave(cpg, 1)
        at = co["a_tg"].repeat_interleave(cpg, 1)
        for _ in range(self.k_diff):
            Hb = Hb + ab * F.conv2d(Hb, self.lap_bg, padding=1, groups=self.n_bg)
            Ht = Ht + at * F.conv2d(Ht, self.lap_tg, padding=1, groups=self.n_tg)
        return Hb, Ht

    # ---------------- 单帧递归 ----------------
    def step(self, z_t, state=None):
        B, C, h, w = z_t.shape
        if state is None:
            state = z_t.new_zeros(B, C, h, w)
        co = self._coefs(z_t)
        Hb, Ht = self._propagate(state, co)

        # 更新: 背景慢 EMA; 目标组 = 背景差分 + 输入依赖门(B2 配方, 无新息门)
        U = self.U(z_t)
        Ub, Ut = U[:, :self.n_bg], U[:, self.n_bg:]
        dec = torch.exp(-co["lam"])
        db, dt_ = dec[:, :self.n_bg], dec[:, self.n_bg:]
        Hb = db * Hb + (1.0 - db) * Ub
        Ut = Ut - self.bg_sub(Hb)
        Ht = dt_ * Ht + (1.0 - dt_) * (co["gate"] * Ut)
        Hn = torch.cat([Hb, Ht], 1)

        v_mean = co["v_tg"].mean(1)
        self.last_v = v_mean.detach()              # M6 跟踪读出
        if self._frames_aux is not None and self.heat_aux:
            self._frames_aux.append((self.heat(Ht), v_mean))
        return z_t + self.out(self.norm(Hn)), Hn

    # ---------------- 前向(clip) ----------------
    def _spatial(self, x):
        if not self.use_ss2d:
            return x
        return x + self.ss2d(self.ln(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

    def forward(self, x):
        # fp16 门控递归易溢出污染 BN → AMP 训练时整块豁免 autocast 内部 fp32;
        # 验证期整模型 .half()(无 autocast)保持原 dtype 直通。
        if torch.is_autocast_enabled():
            with torch.autocast("cuda", enabled=False):
                return self._forward(x.float()).to(x.dtype)
        return self._forward(x)

    def _forward(self, x):
        BT, C, H, W = x.shape
        T = self.T
        assert BT % T == 0, f"batch({BT}) 不能被 nframes({T}) 整除"
        if self.heat_aux and self.training:
            self._frames_aux = []
        z = self._spatial(x).view(BT // T, T, C, H, W)
        state, outs = None, []
        for t in range(T):
            o, state = self.step(z[:, t], state)
            outs.append(o)
        out = torch.stack(outs, 1)
        return out[:, -1] if self.keep == "last" else out.reshape(BT, C, H, W)

    def pop_aux(self):
        aux, self._frames_aux = self._frames_aux, None
        return aux

    @torch.no_grad()
    def forward_stream(self, x_t, state=None):
        """在线推理: 单帧 (B,C,H,W) + 外部 state; last_v 供跟踪器读取。"""
        return self.step(self._spatial(x_t), state)

    # ---------------- 无注入预测(rollout, 批次 7 探针) ----------------
    def _coefs_transport(self, co):
        """系数随流: v_l 自平流(Burgers), α/λ 随组搬运, v_g 冻结。
        搬运为凸组合 → 界与 CFL 保持, 非扩张稳定性在 rollout 中成立。"""
        B, _, h, w = co["v_g"].shape
        v_tgf = co["v_tg"].reshape(B, 2 * self.g_tg, h, w)
        vl = self._advect_by(co["vl"], v_tgf, self.g_tg)
        a_tg = self._advect_by(co["a_tg"], v_tgf, self.g_tg)
        v_bgf = (co["v_g"].unsqueeze(1).expand(B, self.g_bg, 2, h, w)
                 .reshape(B, 2 * self.g_bg, h, w))
        a_bg = self._advect_by(co["a_bg"], v_bgf, self.g_bg)
        lam = self._advect_by(co["lam"], co["v_all"], self.G)
        v_g = co["v_g"]
        v_tg = v_g.unsqueeze(1) + vl.view(B, self.g_tg, 2, h, w)
        v_bg = v_g.unsqueeze(1).expand(B, self.g_bg, 2, h, w)
        v_all = torch.cat([v_bg, v_tg], 1).reshape(B, 2 * self.G, h, w)
        return dict(vl=vl, a_bg=a_bg, a_tg=a_tg, lam=lam, gate=None,
                    v_g=v_g, v_tg=v_tg, v_all=v_all)

    @torch.no_grad()
    def rollout_step(self, state, co):
        Hb, Ht = self._propagate(state, co)
        dec = torch.exp(-co["lam"])
        Hb = dec[:, :self.n_bg] * Hb
        Ht = dec[:, self.n_bg:] * Ht
        out = dict(energy=(Ht.float() ** 2).mean(1),
                   heat=self.heat(Ht) if self.heat_aux else None,
                   v_mean=co["v_tg"].mean(1))
        return torch.cat([Hb, Ht], 1), self._coefs_transport(co), out

    @torch.no_grad()
    def observe_then_rollout(self, x, k):
        """先流式吸收 T 帧观测(逐位复用 step), 再 k 步无注入预测。
        x: 本块输入特征 (B·T,C,h,w), 用 forward pre-hook 从整网截获。"""
        BT, C, h, w = x.shape
        T = self.T
        assert BT % T == 0
        z = self._spatial(x).view(BT // T, T, C, h, w)
        state, co, obs = None, None, []
        for t in range(T):
            z_t = z[:, t]
            co = self._coefs(z_t)
            _, state = self.step(z_t, state)
            Ht = state[:, self.n_bg:]
            obs.append(dict(energy=(Ht.float() ** 2).mean(1),
                            heat=self.heat(Ht) if self.heat_aux else None,
                            v_mean=co["v_tg"].mean(1)))
        pred = []
        for _ in range(k):
            state, co, out = self.rollout_step(state, co)
            pred.append(out)
        return dict(obs=obs, pred=pred, state=state, co=co)


# =============================== 自检 ===============================
if __name__ == "__main__":
    import os
    os.environ.setdefault("PHYS_SCAN_CHUNK", "16")
    torch.manual_seed(0)
    B, T, C, H, W = 2, 3, 16, 20, 20
    x = torch.randn(B * T, C, H, W, requires_grad=True)

    # (a) 各配置前向/反向
    for ss, ego, keep in [(True, True, "last"), (False, False, "all"),
                          (True, False, "last"), (False, True, "all")]:
        m = TraceV2(C, nframes=T, groups=4, use_ss2d=ss, use_ego=ego,
                    keep=keep, heat_aux=True)
        y = m(x)
        y.mean().backward()
        exp = (B, C, H, W) if keep == "last" else (B * T, C, H, W)
        assert y.shape == exp, (ss, ego, keep, y.shape)
    print("config matrix (ss2d×ego×keep): forward/backward ok")

    # (b) 流式 == clip 前向(逐位一致)
    m = TraceV2(C, nframes=T, groups=4, keep="last").eval()
    with torch.no_grad():
        y_clip = m(x.detach())
        state, y_s = None, None
        for t in range(T):
            y_s, state = m.forward_stream(x.detach().view(B, T, C, H, W)[:, t], state)
    print("streaming == clip forward:", bool(torch.allclose(y_clip, y_s, atol=1e-5)))

    # (c) 半精度回归: ego half + 全模块 bf16
    m_h = TraceV2(C, nframes=T, groups=4).eval()
    m_h.ego = m_h.ego.half()
    with torch.no_grad():
        _ = m_h(x.detach())
    m_bf = TraceV2(C, nframes=T, groups=4).eval().bfloat16()
    with torch.no_grad():
        y_bf = m_bf(x.detach().bfloat16())
    print("half-eval regression: ok | bf16 forward:", tuple(y_bf.shape), y_bf.dtype)

    # (d) rollout: 形状/能量衰减/系数界保持
    m_r = TraceV2(C, nframes=T, groups=4, keep="all").eval()
    with torch.no_grad():
        r = m_r.observe_then_rollout(x.detach(), k=6)
    e = [float(p["energy"].sum()) for p in r["pred"]]
    assert r["pred"][0]["energy"].shape == (B, H, W)
    assert e[-1] < e[0], "无注入下目标组能量应整体衰减"
    assert float(r["co"]["a_tg"].max()) <= 0.25 + 1e-5
    assert float(r["co"]["vl"].abs().max()) <= m_r.v_max + 1e-4
    print("rollout: obs", len(r["obs"]), "pred", len(r["pred"]),
          "| energy:", " → ".join(f"{v:.2f}" for v in e))

    # (e) 参数量对比(v2 应显著小于 v1 的 61.6K)
    n64 = sum(p.numel() for p in TraceV2(64).parameters())
    print(f"TraceV2(C=64) params: {n64/1e3:.1f}K  (v1 REF-P3 为 61.6K)")
