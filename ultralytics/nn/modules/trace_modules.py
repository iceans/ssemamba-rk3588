# -*- coding: utf-8 -*-
"""
trace_modules.py — TRACE 框架模块实现（对应 method_design.md 的 M1/M2/M3升级/M4/M5）
================================================================================
与 phys_ss2d.py 的关系: 该文件是增量文件, 复用其中的 SS2D / TemporalUnfold /
TemporalSelect / PhysSS2D(M3 基础版), 新增:

  [M1] SPDConv   : space-to-depth 无损下采样卷积(替换 P1/P2 的 stride-2 Conv)
       HFStem    : 可学习 σ 的 DoG 对比先验旁路 + SPD 下采样 stem(输入端, P1)
  [M2+M3+M4] TraceSS2D : 升级版时序物理块 —— 在 PhysSS2D 基础上加入
       · 参考系分解(M2): 全局仿射速度 v_g(GAP→MLP→6参, 伽利略变换) + 局部残差 v_l;
         状态分组 [背景组|目标组]: 背景组仅随 v_g 输运、长记忆慢更新(≈运动补偿后的
         静态背景模型), 并做可微背景差分 U_tg ← U_tg − W_bg(H̃_bg);
       · 各向异性扩散(M3升级): D = a⊥I + (a∥−a⊥)v̂v̂ᵀ, 沿运动方向的"管状"不确定性;
       · 新息门控(M4): ŷ=C_p H̃, r=U−ŷ, g=σ(MLP([U,ŷ,r])), 自适应卡尔曼增益,
         抑制闪烁噪声、放大跨帧一致证据;
       · 场读出辅助(M5 的模块侧): 逐帧热图 logits 头 + 速度场缓存(pop_aux 取用),
         推理时暴露 last_v 供状态域跟踪(M6 v1)读取。
  [M5] 损失工具: gaussian_heatmap / heatmap_focal_loss / velocity_loss
       (数据侧 GT 格式见各函数 docstring, 集成示意见文件末尾注释)。

--------------------------------------------------------------------------------
Ultralytics 集成(在 phys_ss2d.py 已注册的基础上追加):
  1) 本文件放到 ultralytics/nn/modules/trace_modules.py
  2) tasks.py 顶部追加:
        from ultralytics.nn.modules.trace_modules import TraceSS2D, SPDConv, HFStem
  3) parse_model() 的 c2 判断链追加(注意 SPDConv/HFStem 要做宽度缩放,
     与官方 Conv 分支保持一致):
        elif m in {SPDConv, HFStem}:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is TraceSS2D:
            c2 = ch[f]
            args = [ch[f], *args]
  4) 模型配置见 yolov8-trace.yaml(P2 检测头 + P2/P3 双时序块)。

TraceSS2D 位置参数(yaml 顺序):
  [nframes, d_state, expand, groups, v_max, keep, diff_substeps, bg_ratio,
   use_ss2d, use_ego, use_innov, aniso, use_advection, use_diffusion, heat_aux]
消融对照(论文表直接映射):
  对角 A 时序 Mamba : use_advection=False, use_diffusion=False, use_ego=False
  无参考系分解(M2)  : use_ego=False        无新息门控(M4): use_innov=False
  各向同性扩散      : aniso=False           无空间扫描    : use_ss2d=False
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                        # 包内使用
    from .phys_ss2d import SS2D
except ImportError:                         # 独立自检
    from phys_ss2d import SS2D


# ============================================================================
# 通用小件
# ============================================================================
class _ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================================
# M1 高频证据编码
# ============================================================================
class SPDConv(nn.Module):
    """space-to-depth 下采样卷积: (B,C,H,W) → unshuffle×2 → (B,4C,H/2,W/2) → conv → c2。
    高频信息进通道而非被 stride 丢弃, 针对 3×3~9×9 目标的能量湮灭问题。"""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = _ConvBNAct(4 * c1, c2, k, 1)

    def forward(self, x):
        return self.conv(F.pixel_unshuffle(x, 2))


class DoG(nn.Module):
    """可学习 σ 的高斯差分组(LoG/局部对比先验的可微化)。输入 (B,C,H,W),
    输出 (B, C*len(scales), H, W): d_k = G(σ_k)*x − G(1.6σ_k)*x。"""

    def __init__(self, c_in=1, scales=(1.0, 2.0, 4.0), k=9):
        super().__init__()
        self.c_in, self.k = c_in, k
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(list(scales))))

    def _kernel(self, sigma):                          # (k,k), 归一化, 可反传到 σ
        x = torch.arange(self.k, device=sigma.device, dtype=torch.float32) - (self.k - 1) / 2
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return torch.outer(g, g)

    def forward(self, x):
        assert x.shape[1] == self.c_in, (
            f"[DoG/HFStem] 构建时 c_in={self.c_in}, 实际输入 {x.shape[1]} 通道。"
            f"通常原因: tasks.py 中 TemporalUnfold 的注册仍是旧版 'c2 = 1', "
            f"需改为 'c2 = args[2] if len(args) > 2 else 1' 后重新构建模型。")
        outs = []
        for i in range(self.log_sigma.numel()):
            s = self.log_sigma[i].exp()
            kd = (self._kernel(s) - self._kernel(1.6 * s))          # DoG 核
            w = kd.expand(self.c_in, 1, self.k, self.k).to(x.dtype)
            outs.append(F.conv2d(x, w, padding=self.k // 2, groups=self.c_in))
        return torch.cat(outs, 1)


class HFStem(nn.Module):
    """P1 stem(stride 2): [x ‖ DoG(x)] → space-to-depth ×2 → conv → c2。
    对比先验通道让 SS2D/检测头从第一层就拿到"点状高频候选"证据。"""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.dog = DoG(c1)
        self.conv = _ConvBNAct(4 * (c1 + 3 * c1), c2, k, 1)

    def forward(self, x):
        x = torch.cat([x, self.dog(x)], 1)
        return self.conv(F.pixel_unshuffle(x, 2))


# ============================================================================
# M2 + M3升级 + M4: TraceSS2D
# ============================================================================
class TraceSS2D(nn.Module):
    """升级版时序物理块。输入 (B·T, C, H, W); keep='all' 同形状, 'last' → (B,C,H,W)。

    状态分组: groups 个速度组中, round(bg_ratio·groups) 个为背景组(仅 v_g 输运,
    长记忆), 其余为目标组(v_g+v_l 输运, 各向异性扩散, 新息门控)。
    训练时逐帧缓存热图 logits 与速度场(pop_aux() 取用, 见 M5 损失工具);
    推理时 last_v 始终可读, 供 M6 状态域跟踪使用。
    """

    def __init__(self, c1, nframes=5, d_state=16, expand=1.0, groups=4,
                 v_max=3.0, keep="last", diff_substeps=2, bg_ratio=0.25,
                 use_ss2d=True, use_ego=True, use_innov=True, aniso=True,
                 use_advection=True, use_diffusion=True, heat_aux=True):
        super().__init__()
        assert keep in ("all", "last") and c1 % groups == 0
        self.T, self.G, self.n = nframes, groups, c1
        self.v_max, self.k_diff, self.a_cfl = v_max, diff_substeps, 0.25
        self.keep = keep
        self.use_ss2d, self.use_ego, self.use_innov = use_ss2d, use_ego, use_innov
        self.aniso, self.use_adv, self.use_dif = aniso, use_advection, use_diffusion
        self.heat_aux = heat_aux

        cpg = c1 // groups                              # 每组通道数
        self.g_bg = max(1, round(bg_ratio * groups))
        self.g_tg = groups - self.g_bg
        assert self.g_tg >= 1, "bg_ratio 过大, 目标组至少保留 1 组"
        self.n_bg, self.n_tg = self.g_bg * cpg, self.g_tg * cpg

        # ---- 帧内空间: VSS 残差 ----
        if use_ss2d:
            self.ln = nn.LayerNorm(c1)
            self.ss2d = SS2D(c1, d_state=d_state, expand=expand)

        # ---- 系数头: [v_l | α_bg | α_tg(各向异性时为 a∥,a⊥) | λ | (gate)] ----
        self.n_vl = 2 * self.g_tg
        self.n_atg = 2 * self.g_tg if aniso else self.g_tg
        self.n_gate = 0 if use_innov else self.n_tg
        n_coef = self.n_vl + self.g_bg + self.n_atg + c1 + self.n_gate
        self.coef = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1), nn.SiLU(),
            nn.Conv2d(c1, n_coef, 1))

        # ---- M2: 全局仿射速度头(伽利略参考系变换) ----
        if use_ego:
            self.ego = nn.Sequential(nn.Linear(c1, c1 // 2), nn.SiLU(),
                                     nn.Linear(c1 // 2, 6))
            nn.init.zeros_(self.ego[-1].weight)
            nn.init.zeros_(self.ego[-1].bias)           # 初始 v_g = 0

        # ---- 源项 / 背景差分 / 输出 ----
        self.U = nn.Conv2d(c1, c1, 1)
        self.bg_sub = nn.Conv2d(self.n_bg, self.n_tg, 1)      # 可微背景差分
        nn.init.zeros_(self.bg_sub.weight); nn.init.zeros_(self.bg_sub.bias)
        self.Cgate = nn.Conv2d(c1, c1, 1)
        self.norm = nn.GroupNorm(1, c1)
        self.out = nn.Conv2d(c1, c1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 块=恒等

        # ---- M4: 新息门控(自适应卡尔曼增益) ----
        if use_innov:
            self.pred_meas = nn.Conv2d(self.n_tg, self.n_tg, 1)        # ŷ = C_p H̃
            self.gate = nn.Sequential(
                nn.Conv2d(3 * self.n_tg, self.n_tg, 1), nn.SiLU(),
                nn.Conv2d(self.n_tg, self.n_tg, 3, padding=1, groups=self.n_tg),
                nn.SiLU(), nn.Conv2d(self.n_tg, self.n_tg, 1))

        # ---- M5(模块侧): 热图辅助头 ----
        if heat_aux:
            self.heat = nn.Conv2d(self.n_tg, 1, 1)
            nn.init.constant_(self.heat.bias, -4.0)     # 稀疏正样本先验

        # ---- 系数头初始化: 恒等 + 轻微 EMA 起步 ----
        final = self.coef[-1]
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
        with torch.no_grad():
            o = self.n_vl
            final.bias[o:o + self.g_bg + self.n_atg] = -2.0            # α 小
            o += self.g_bg + self.n_atg
            final.bias[o:o + self.n_bg] = -3.9          # 背景组 e^{−λ} ≈ 0.98
            final.bias[o + self.n_bg:o + c1] = -2.2     # 目标组 e^{−λ} ≈ 0.90

        # ---- 扩散模板(固定 buffer) ----
        def rep(k, n): return k.view(1, 1, 3, 3).repeat(n, 1, 1, 1)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        kxx = torch.tensor([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]])
        kxy = 0.25 * torch.tensor([[1., 0., -1.], [0., 0., 0.], [-1., 0., 1.]])
        self.register_buffer("lap_bg", rep(lap, self.n_bg))
        if aniso:
            self.register_buffer("kxx", rep(kxx, self.n_tg))
            self.register_buffer("kyy", rep(kxx.t(), self.n_tg))
            self.register_buffer("kxy", rep(kxy, self.n_tg))
        else:
            self.register_buffer("lap_tg", rep(lap, self.n_tg))

        self._grid = {}
        self._frames_aux = None
        self.last_v = None                              # (B,2,h,w) 供 M6 跟踪读取

    # ---------------- 网格缓存(含跨 save/load 的 device 修复) ----------------
    def _base_grid(self, h, w, dev):
        dev = torch.device(dev)
        key = (h, w, str(dev))
        cached = self._grid.get(key)
        if cached is None or cached.device != dev:      # 校验真实 device
            yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                    torch.arange(w, device=dev, dtype=torch.float32),
                                    indexing="ij")
            cached = torch.stack((xx, yy), -1)
            self._grid[key] = cached
        return cached

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_grid"], state["_frames_aux"], state["last_v"] = {}, None, None
        return state

    def _advect(self, H, v_all):
        """半拉格朗日; H: (B,N,h,w), v_all: (B,2G,h,w) 按组配对 [vx,vy]。"""
        B, N, h, w = H.shape
        G, cpg = self.G, N // self.G
        Hg = H.float().view(B * G, cpg, h, w)
        vg = v_all.float().view(B * G, 2, h, w).permute(0, 2, 3, 1)
        src = self._base_grid(h, w, H.device) - vg
        gx = src[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
        gy = src[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        out = F.grid_sample(Hg, torch.stack((gx, gy), -1), mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.view(B, N, h, w).to(H.dtype)

    # ---------------- 单帧递归 ----------------
    def step(self, z_t, state=None):
        B, C, h, w = z_t.shape
        cpg = C // self.G
        if state is None:
            state = z_t.new_zeros(B, C, h, w)

        coef = self.coef(z_t)
        o = 0
        vl = self.v_max * torch.tanh(coef[:, o:o + self.n_vl]); o += self.n_vl
        a_bg = self.a_cfl * torch.sigmoid(coef[:, o:o + self.g_bg]); o += self.g_bg
        a_tg = self.a_cfl * torch.sigmoid(coef[:, o:o + self.n_atg]); o += self.n_atg
        lam = F.softplus(coef[:, o:o + C]); o += C
        gate_plain = torch.sigmoid(coef[:, o:]) if self.n_gate else None

        # ---- M2: 速度合成 v = v_g + v_l ----
        base = self._base_grid(h, w, z_t.device)
        xn = base[..., 0] * (2.0 / max(w - 1, 1)) - 1.0                 # [-1,1]
        yn = base[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        if self.use_ego:
            # 兼容三种运行态: 纯 fp32 / AMP autocast 训练 / 验证期整模型 .half()。
            # ultralytics 在 trainer.amp=True 时, 训练中的验证会把模型转半精度且
            # 无 autocast 上下文——此处若强制 .float() 会与 Half 权重相乘而崩溃,
            # 故输入跟随权重 dtype, 输出转 fp32 与网格坐标(float32)做仿射运算。
            w_dtype = self.ego[0].weight.dtype
            p = self.ego(z_t.mean((2, 3)).to(w_dtype)).float()          # (B,6)
            vgx = p[:, 0, None, None] * xn + p[:, 1, None, None] * yn + p[:, 2, None, None]
            vgy = p[:, 3, None, None] * xn + p[:, 4, None, None] * yn + p[:, 5, None, None]
            v_g = self.v_max * torch.tanh(torch.stack([vgx, vgy], 1)).to(z_t.dtype)
        else:
            v_g = z_t.new_zeros(B, 2, h, w)
        v_tg = v_g.unsqueeze(1) + vl.view(B, self.g_tg, 2, h, w)        # 目标组
        v_bg = v_g.unsqueeze(1).expand(B, self.g_bg, 2, h, w)           # 背景组
        v_all = torch.cat([v_bg, v_tg], 1).reshape(B, 2 * self.G, h, w)

        # ---- 1) 对流 ----
        H = self._advect(state, v_all) if self.use_adv else state
        Hb, Ht = H[:, :self.n_bg], H[:, self.n_bg:]

        # ---- 2) 扩散(背景各向同性; 目标组可选各向异性 D=a⊥I+(a∥−a⊥)v̂v̂ᵀ) ----
        if self.use_dif:
            ab = a_bg.repeat_interleave(cpg, 1)
            for _ in range(self.k_diff):
                Hb = Hb + ab * F.conv2d(Hb, self.lap_bg, padding=1, groups=self.n_bg)
            if self.aniso:
                a_par, a_perp = a_tg[:, :self.g_tg], a_tg[:, self.g_tg:]
                vv = v_tg / (v_tg.norm(dim=2, keepdim=True) + 1e-3)     # 平滑退化
                vx, vy = vv[:, :, 0], vv[:, :, 1]                       # (B,g_tg,h,w)
                Dxx = (a_perp + (a_par - a_perp) * vx * vx).repeat_interleave(cpg, 1)
                Dyy = (a_perp + (a_par - a_perp) * vy * vy).repeat_interleave(cpg, 1)
                Dxy = ((a_par - a_perp) * vx * vy).repeat_interleave(cpg, 1)
                for _ in range(self.k_diff):            # a∥+a⊥ ≤ 0.5/子步, 显式稳定
                    Ht = (Ht + Dxx * F.conv2d(Ht, self.kxx, padding=1, groups=self.n_tg)
                             + Dyy * F.conv2d(Ht, self.kyy, padding=1, groups=self.n_tg)
                             + 2 * Dxy * F.conv2d(Ht, self.kxy, padding=1, groups=self.n_tg))
            else:
                at = a_tg.repeat_interleave(cpg, 1)
                for _ in range(self.k_diff):
                    Ht = Ht + at * F.conv2d(Ht, self.lap_tg, padding=1, groups=self.n_tg)

        # ---- 3) 更新: 背景慢 EMA; 目标组 = 背景差分 + 新息门控 ----
        U = self.U(z_t)
        Ub, Ut = U[:, :self.n_bg], U[:, self.n_bg:]
        dec = torch.exp(-lam)
        db, dt_ = dec[:, :self.n_bg], dec[:, self.n_bg:]
        Hb = db * Hb + (1.0 - db) * Ub                                   # 背景模型
        Ut = Ut - self.bg_sub(Hb)                                        # 可微扣底
        if self.use_innov:
            yhat = self.pred_meas(Ht)
            g = torch.sigmoid(self.gate(torch.cat([Ut, yhat, Ut - yhat], 1)))
        else:
            g = gate_plain
        Ht = dt_ * Ht + (1.0 - dt_) * (g * Ut)
        Hn = torch.cat([Hb, Ht], 1)

        # ---- M5/M6 读出 ----
        v_mean = v_tg.mean(1)                                            # (B,2,h,w)
        self.last_v = v_mean.detach()
        if self._frames_aux is not None and self.heat_aux:
            self._frames_aux.append((self.heat(Ht), v_mean))

        y = self.norm(F.silu(self.Cgate(z_t)) * Hn)
        return z_t + self.out(y), Hn

    # ---------------- 前向 ----------------
    def _spatial(self, x):
        if not self.use_ss2d:
            return x
        return x + self.ss2d(self.ln(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

    def forward(self, x):                                # x: (B·T, C, H, W)
        # Mamba 系门控乘法(y·SiLU(z))与跨帧递归状态在 fp16 下易溢出(社区已知,
        # 官方 Mamba 建议 bf16)。AMP 训练时本模块整体豁免 autocast、以 fp32 运行
        # (通道数小, 开销可忽略); 验证期整模型 .half() 时(无 autocast 上下文)
        # 保持原 dtype 直通, 与半精度评测兼容。
        if torch.is_autocast_enabled():
            with torch.autocast("cuda", enabled=False):
                return self._forward(x.float()).to(x.dtype)
        return self._forward(x)

    def _forward(self, x):
        BT, C, H, W = x.shape
        T = self.T
        assert BT % T == 0, f"batch({BT}) 不能被 nframes({T}) 整除"
        z = self._spatial(x).view(BT // T, T, C, H, W)
        self._frames_aux = [] if (self.training and self.heat_aux) else None
        state, outs = None, []
        for t in range(T):
            o, state = self.step(z[:, t], state)
            outs.append(o)
        if self._frames_aux:
            heat = torch.stack([a[0] for a in self._frames_aux], 1)     # (B,T,1,h,w)
            vel = torch.stack([a[1] for a in self._frames_aux], 1)      # (B,T,2,h,w)
            self._aux = {"heat": heat, "vel": vel}
        self._frames_aux = None
        out = torch.stack(outs, 1)
        return out[:, -1] if self.keep == "last" else out.reshape(BT, C, H, W)

    def pop_aux(self):
        """训练时由自定义 Loss 调用, 取走逐帧热图 logits 与速度场并清空。"""
        aux = getattr(self, "_aux", None)
        self._aux = None
        return aux

    @torch.no_grad()
    def forward_stream(self, x_t, state=None):
        """在线推理: 单帧 (B,C,H,W) + 外部 state; last_v 可供跟踪器读取。"""
        return self.step(self._spatial(x_t), state)


# ============================================================================
# M5 损失工具(数据侧集成见文末注释)
# ============================================================================
def gaussian_heatmap(hw, centers, sigma=1.5, device="cpu"):
    """生成单帧 GT 热图 (1,h,w)。centers: (n,2) 特征尺度下的 (cx,cy);
    多目标取逐像素 max。tiny 目标建议固定 σ≈1.5(特征像素)。"""
    h, w = hw
    hm = torch.zeros(1, h, w, device=device)
    if centers is None or len(centers) == 0:
        return hm
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32),
                            indexing="ij")
    for cx, cy in centers:
        hm[0] = torch.maximum(hm[0], torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                               / (2 * sigma ** 2)))
    return hm


def heatmap_focal_loss(logits, gt, alpha=2.0, beta=4.0):
    """CenterNet 式 penalty-reduced focal loss。logits/gt: (...,1,h,w), gt∈[0,1]。"""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.999)
    loss_pos = -(((1 - p) ** alpha) * p.log())[pos].sum()
    loss_neg = -(((1 - gt) ** beta) * (p ** alpha) * (1 - p).log())[~pos].sum()
    return (loss_pos + loss_neg) / pos.sum().clamp(min=1)


def velocity_loss(v_pred, centers, disp_gt, radius=1):
    """在目标中心邻域监督速度场。v_pred: (2,h,w); centers: (n,2) 特征坐标;
    disp_gt: (n,2) 该帧→下一帧的位移(特征尺度像素, 即 GT 轨迹差分 / stride)。"""
    if centers is None or len(centers) == 0:
        return v_pred.sum() * 0.0
    h, w = v_pred.shape[-2:]
    loss, cnt = 0.0, 0
    for (cx, cy), d in zip(centers, disp_gt):
        x0, x1 = max(0, int(cx) - radius), min(w, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(h, int(cy) + radius + 1)
        patch = v_pred[:, y0:y1, x0:x1]
        tgt = torch.as_tensor(d, device=v_pred.device, dtype=v_pred.dtype)
        loss = loss + F.smooth_l1_loss(patch, tgt.view(2, 1, 1).expand_as(patch))
        cnt += 1
    return loss / max(cnt, 1)


# ----------------------------------------------------------------------------
# 训练集成示意(放入自定义 Loss / trainer, 伪代码):
#
#   trace_blocks = [m for m in de_parallel(model).modules() if isinstance(m, TraceSS2D)]
#   loss_aux = 0
#   for blk in trace_blocks:
#       aux = blk.pop_aux()
#       if aux is None: continue
#       B, T = aux["heat"].shape[:2]; h, w = aux["heat"].shape[-2:]
#       stride = imgsz / w                              # 该块所在层的下采样率
#       for b in range(B):
#           for t in range(T):
#               ctr  = frame_centers[b][t] / stride     # 数据侧: 逐帧中心(图像坐标)
#               disp = frame_disp[b][t] / stride        # 数据侧: 逐帧位移(轨迹差分)
#               gt   = gaussian_heatmap((h, w), ctr, 1.5, aux["heat"].device)
#               loss_aux += lambda_h * heatmap_focal_loss(aux["heat"][b, t], gt)
#               loss_aux += lambda_v * velocity_loss(aux["vel"][b, t], ctr, disp)
#   total_loss = det_loss + loss_aux / max(len(trace_blocks), 1)
#
# 数据侧需要提供: frame_centers(逐帧目标中心)与 frame_disp(相邻帧中心差分);
# 二者都能从你的时序 dataset 的逐帧标注/轨迹 ID 直接算出。keep='last' 时检测损失
# 只用末帧, 但热图/速度监督仍作用于全部 T 帧——这正是"状态场深监督"的意义。
# DDP 下请通过 de_parallel(model) 访问模块实例。
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)
    os.environ["PHYS_SCAN_CHUNK"] = "16"

    # M1 形状检查
    x_img = torch.randn(4, 1, 64, 64)
    print("HFStem :", tuple(HFStem(1, 16)(x_img).shape))      # (4,16,32,32)
    print("SPDConv:", tuple(SPDConv(16, 32)(torch.randn(4, 16, 32, 32)).shape))

    # TraceSS2D 开关矩阵: 前向/反向
    B, T, C, H, W = 2, 3, 16, 20, 20
    x = torch.randn(B * T, C, H, W)
    for ego in (True, False):
        for innov in (True, False):
            for an in (True, False):
                m = TraceSS2D(C, nframes=T, groups=4, keep="all",
                              use_ego=ego, use_innov=innov, aniso=an)
                m.train()
                y = m(x.requires_grad_(True))
                y.mean().backward()
                aux = m.pop_aux()
                assert aux["heat"].shape == (B, T, 1, H, W)
                assert aux["vel"].shape == (B, T, 2, H, W)
    print("flag matrix (ego×innov×aniso): forward/backward/aux ok")

    # keep='last' + 流式一致性(eval)
    m = TraceSS2D(C, nframes=T, groups=4, keep="all").eval()
    with torch.no_grad():
        y_clip = m(x)
        state, outs = None, []
        for t in range(T):
            o, state = m.forward_stream(x.view(B, T, C, H, W)[:, t], state)
            outs.append(o)
        y_str = torch.stack(outs, 1).reshape(B * T, C, H, W)
    print("streaming == clip forward:", torch.allclose(y_clip, y_str, atol=1e-4))
    print("last_v shape:", tuple(m.last_v.shape))

    # (f) 半精度回归测试: 模拟 ultralytics amp=True 时验证期的"裸半精度模型"
    #     (整模型 .half()/.bfloat16() 且无 autocast 上下文)
    m_h = TraceSS2D(C, nframes=T, groups=4, keep="last").eval()
    m_h.ego = m_h.ego.half()                       # 只把 ego 转半精度: 定点复现旧崩溃
    with torch.no_grad():
        _ = m_h(x.detach())
    m_bf = TraceSS2D(C, nframes=T, groups=4, keep="last").eval().bfloat16()
    with torch.no_grad():
        y_bf = m_bf(x.detach().bfloat16())
    print("half-eval regression:", "ok |", "bf16 full forward:", tuple(y_bf.shape),
          y_bf.dtype)

    # M5 损失冒烟测试
    hm_gt = gaussian_heatmap((H, W), [(5.0, 5.0), (12.0, 8.0)], 1.5)
    logits = torch.randn(1, H, W, requires_grad=True)
    l1 = heatmap_focal_loss(logits, hm_gt)
    l2 = velocity_loss(torch.randn(2, H, W, requires_grad=True),
                       [(5.0, 5.0)], [(1.0, -0.5)])
    (l1 + l2).backward()
    print(f"aux losses finite: heat={l1.item():.3f}, vel={l2.item():.3f}")
    n_params = sum(p.numel() for p in TraceSS2D(64, nframes=5).parameters())
    print(f"TraceSS2D(C=64) params: {n_params/1e3:.1f}K")
