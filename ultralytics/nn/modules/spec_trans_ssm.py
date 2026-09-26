# -*- coding: utf-8 -*-
"""
SpecTransSSM: 频域选择性输运状态空间模块 (论点 a 的实现)
=========================================================

核心数学事实
------------
若输运系数 (v_t, D_t, λ_t) **空间均匀但逐帧时变**(输入依赖), 则每步状态转移
    H_t = exp(Δ·L_t) H_{t-1} + B u_t,   L_t = -(v_t·∇) + ∇·(D_t∇) - λ_t
的转移算子是傅里叶乘子, 其符号为
    a_t(k) = exp( -i(k·v_t) - kᵀD_t k - λ_t )        (Δ 吸收进系数)
所有傅里叶乘子被 2D DFT 同时对角化且互相交换, 因此在频域中这**字面上**就是
Mamba 式逐频率复对角选择性扫描:
    ĥ_t(k) = a_t(k) ĥ_{t-1}(k) + b̂_t(k)
性质(代码中均有断言/测试):
  * 相位 = 平移 (Fourier shift theorem), 亚像素精确, 无插值模糊;
  * D 半正定 (LLᵀ 参数化) 且 λ≥0  ⇒  |a_t(k)| ≤ 1 逐频率成立
    ⇒ **无条件稳定, 不存在 CFL 约束** (对比显式差分版 phys_ss2d.py 的 CFL 子步);
  * 时间维支持 Kogge-Stone 并行前缀扫描 (log T 步) —— "这仍然是 SSM" 的字面依据。

逐像素系数是不可并行的墙 (平移不变性破坏 ⇒ 傅里叶对角化失效), 因此本文件同时
提供局部路径 LocalTransport: 逐像素 v_l, 半拉格朗日 backward warp (grid_sample),
序贯递归但同样无条件稳定 (双线性插值是凸组合, 最大范数不增)。

mode:
  'spectral' : 全局谱域路径 (可并行扫描)                          —— 论点 a
  'local'    : 逐像素半拉格朗日路径 (序贯)                        —— 论点 b 的局部臂
  'hybrid'   : 状态通道分组 [bg=谱域全局 | tg=局部 v_g+v_l]       —— 论点 b / M2
  'diag'     : use_adv=use_diff=False 的谱域路径 = 逐像素对角 A    —— 论点 c 的对照

门控与并行性的互斥 (论文中要正面写出的诚实结论):
  gate='innov' (完整新息门, 依赖演化中的状态) 会破坏扫描结合律, 只能序贯;
  gate='input' (输入代理门, 用 u_t 与帧差 u_t-u_{t-1}) 保持并行扫描。
  这本身构成一个消融轴 (E6)。

Ultralytics 集成: 见文件末尾 register_trace_modules() 与 tasks.py 补丁说明。
状态: 已通过 smoke_test.py 的 7 项测试 (CPU, torch>=2.x)。未做大规模训练验证。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpecTransSSM", "SPDConv", "ClipToBatch", "BatchToClip",
           "register_trace_modules"]


# ----------------------------------------------------------------------------
# 并行线性递归扫描 (Kogge-Stone / doubling): h_t = a_t h_{t-1} + b_t
# a, b: complex tensor, shape (B, T, ...)
# ----------------------------------------------------------------------------
def linear_scan_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    T = a.shape[1]
    offset = 1
    while offset < T:
        a_new = a.clone()
        b_new = b.clone()
        b_new[:, offset:] = b[:, offset:] + a[:, offset:] * b[:, :-offset]
        a_new[:, offset:] = a[:, offset:] * a[:, :-offset]
        a, b = a_new, b_new
        offset *= 2
    return b  # b[:, t] == h_t


def linear_scan_sequential(a: torch.Tensor, b: torch.Tensor,
                           h0: torch.Tensor | None = None) -> torch.Tensor:
    T = a.shape[1]
    hs = []
    h = h0 if h0 is not None else torch.zeros_like(b[:, 0])
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


# ----------------------------------------------------------------------------
# 全局系数头: 逐帧、逐状态通道预测 (vx, vy, l11, l21, l22, lam)
#   selective=True  : 由输入 GAP 特征回归 (输入依赖, S6 式)
#   selective=False : 可学习常数 (LTI, ConvS5 类比, 消融 E3 用)
# D = LLᵀ (Cholesky) 保证半正定; v = v_max·tanh(·); λ = softplus(·) ≥ 0
# 末层零初始化 ⇒ 初始 v=0、D≈0 ⇒ 模块初始等价于 'diag', 支持热插拔两阶段训练。
# ----------------------------------------------------------------------------
class GlobalCoeffHead(nn.Module):
    N_COEF = 6

    def __init__(self, d_state: int, v_max: float = 4.0, selective: bool = True,
                 hidden: int = 64):
        super().__init__()
        self.d, self.v_max, self.selective = d_state, v_max, selective
        if selective:
            self.mlp = nn.Sequential(
                nn.Linear(d_state, hidden), nn.SiLU(),
                nn.Linear(hidden, d_state * self.N_COEF),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.const = nn.Parameter(torch.zeros(d_state * self.N_COEF))
        # 偏置: softplus(-4)≈0.018 → 初始扩散/衰减都很小
        self.register_buffer("bias", torch.tensor(
            [0.0, 0.0, -4.0, -4.0, -4.0, -2.0]).repeat(d_state))

    def forward(self, u: torch.Tensor, B: int, T: int):
        """u: (B*T, d, H, W) → 6 组系数, 各 (B, T, d, 1, 1)"""
        if self.selective:
            # 统计量用 fp32 求 (数值稳), 进 MLP 前转成权重 dtype (兼容 model.half()),
            # 出来立刻回 fp32 —— 系数必须 fp32, 因为要构造复数乘子
            desc = u.mean(dim=(2, 3), dtype=torch.float32)
            w_dtype = self.mlp[0].weight.dtype
            z = self.mlp(desc.to(w_dtype)).float() + self.bias.float()
        else:
            z = (self.const.float() + self.bias.float()).expand(B * T, -1)
        z = z.view(B, T, self.d, self.N_COEF)
        vx = self.v_max * torch.tanh(z[..., 0])
        vy = self.v_max * torch.tanh(z[..., 1])
        l11 = F.softplus(z[..., 2])
        l21 = z[..., 3] * 0.1                              # 非对角项小尺度
        l22 = F.softplus(z[..., 4])
        lam = F.softplus(z[..., 5])
        return [c[..., None, None] for c in (vx, vy, l11, l21, l22, lam)]


# ----------------------------------------------------------------------------
# 谱域全局输运 (论点 a 核心)
# ----------------------------------------------------------------------------
class SpectralTransport(nn.Module):
    def __init__(self, d_state: int, v_max: float = 4.0, selective: bool = True,
                 use_adv: bool = True, use_diff: bool = True, use_decay: bool = True,
                 pad: int = 8, scan: str = "parallel"):
        super().__init__()
        self.d = d_state
        self.use_adv, self.use_diff, self.use_decay = use_adv, use_diff, use_decay
        self.pad, self.scan = pad, scan
        self.coeff = GlobalCoeffHead(d_state, v_max, selective)
        self._freq_cache = {}

    def _freqs(self, Hp, Wp, device):
        # 按 (尺寸) 缓存, 但每次取用都 .to(device): warmup 的 CPU 假输入与
        # 训练的 cuda 输入共用一个 module, 缓存里存哪个 device 都会撞另一个
        key = (Hp, Wp)
        if key not in self._freq_cache:
            wy = 2 * math.pi * torch.fft.fftfreq(Hp)
            wx = 2 * math.pi * torch.fft.rfftfreq(Wp)
            self._freq_cache[key] = (wy.view(1, 1, 1, Hp, 1),
                                     wx.view(1, 1, 1, 1, -1))
        wy, wx = self._freq_cache[key]
        return wy.to(device), wx.to(device)

    def transition(self, u: torch.Tensor, B: int, T: int, Hp: int, Wp: int):
        """构造逐频率复对角转移 a_t(k), 形状 (B,T,d,Hp,Wr) complex64"""
        vx, vy, l11, l21, l22, lam = self.coeff(u, B, T)
        wy, wx = self._freqs(Hp, Wp, u.device)
        real = torch.zeros(B, T, self.d, Hp, wx.shape[-1], device=u.device)
        imag = torch.zeros_like(real)
        if self.use_adv:
            imag = -(vx * wx + vy * wy)                    # 相位 = 平移
        if self.use_diff:
            real = real - ((l11 * wx + l21 * wy) ** 2 + (l22 * wy) ** 2)
        if self.use_decay:
            real = real - lam
        a = torch.exp(torch.complex(real, imag))
        # 稳定性由构造保证: real ≤ 0 ⇒ |a| ≤ 1, 无 CFL
        return a, (vx, vy)

    def forward(self, inj: torch.Tensor, u: torch.Tensor, B: int, T: int,
                h0: torch.Tensor | None = None):
        """inj: 门控后的注入量 (B*T, d, H, W); 返回像素域状态 (B*T,d,H,W) 与末帧频域状态"""
        N, d, H, W = inj.shape
        p = self.pad
        x = F.pad(inj, (p, p, p, p), mode="reflect") if p > 0 else inj
        Hp, Wp = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b = torch.fft.rfft2(x, norm="ortho").view(B, T, d, Hp, -1)
            a, v_global = self.transition(u, B, T, Hp, Wp)
            if h0 is not None:
                b = b.clone()
                b[:, 0] = b[:, 0] + a[:, 0] * h0
            if self.scan == "parallel":
                h_hat = linear_scan_parallel(a, b)
            else:
                h_hat = linear_scan_sequential(a, b)
            h = torch.fft.irfft2(h_hat.reshape(N, d, Hp, -1), s=(Hp, Wp),
                                 norm="ortho")
        if p > 0:
            h = h[..., p:-p, p:-p]
        return h, h_hat[:, -1], v_global


# ----------------------------------------------------------------------------
# 局部半拉格朗日输运 (逐像素 v_l, 序贯, 无条件稳定)
# ----------------------------------------------------------------------------
class LocalTransport(nn.Module):
    def __init__(self, d_state: int, v_max_local: float = 3.0,
                 innov_gate: bool = False):
        super().__init__()
        self.d, self.v_max = d_state, v_max_local
        self.innov = innov_gate
        if innov_gate:
            # M4: 预测量测头 C_p 与增益网络 g = sigma(MLP([u, y_hat, r]))
            self.pred_head = nn.Conv2d(d_state, d_state, 1)
            self.gnet = nn.Sequential(
                nn.Conv2d(3 * d_state, d_state, 1), nn.SiLU(),
                nn.Conv2d(d_state, d_state, 3, padding=1, groups=d_state),
                nn.SiLU(), nn.Conv2d(d_state, d_state, 1))
        self.v_head = nn.Sequential(
            nn.Conv2d(d_state, d_state, 3, padding=1, groups=d_state), nn.SiLU(),
            nn.Conv2d(d_state, 2, 1),
        )
        nn.init.zeros_(self.v_head[-1].weight)
        nn.init.zeros_(self.v_head[-1].bias)
        self.log_lam = nn.Parameter(torch.full((1, d_state, 1, 1), -2.0))

    @staticmethod
    def _warp(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """backward semi-Lagrangian: 在出发点 p - v 采样 h (内部 fp32, 出口还原)"""
        dt = h.dtype
        B, _, H, W = h.shape
        ys = torch.linspace(-1, 1, H, device=h.device)
        xs = torch.linspace(-1, 1, W, device=h.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        v32 = v.float()
        vn = torch.stack((2 * v32[:, 0] / max(W - 1, 1),
                          2 * v32[:, 1] / max(H - 1, 1)), dim=-1)
        return F.grid_sample(h.float(), base - vn, mode="bilinear",
                             padding_mode="border",
                             align_corners=True).to(dt)

    def forward(self, inj: torch.Tensor, u: torch.Tensor, B: int, T: int,
                v_global=None, h0: torch.Tensor | None = None):
        N, d, H, W = inj.shape
        inj = inj.view(B, T, d, H, W)
        u = u.view(B, T, d, H, W)
        decay = torch.exp(-F.softplus(self.log_lam))
        h = h0 if h0 is not None else torch.zeros(B, d, H, W, device=inj.device)
        outs = []
        for t in range(T):
            v = self.v_max * torch.tanh(self.v_head(u[:, t]))     # (B,2,H,W)
            if v_global is not None:  # 伽利略分解: v = v_g + v_l
                vgx, vgy = v_global
                v = v + torch.cat([vgx[:, t, :1].expand(-1, -1, H, W),
                                   vgy[:, t, :1].expand(-1, -1, H, W)],
                                  dim=1).to(v.dtype)
            h_tilde = self._warp(h, v) * decay          # 预测步
            if self.innov:                                # 量测更新步 (卡尔曼增益)
                y_hat = self.pred_head(h_tilde)
                r = u[:, t] - y_hat                       # 新息
                g = torch.sigmoid(self.gnet(
                    torch.cat([u[:, t], y_hat, r], dim=1)))
                h = h_tilde + g * u[:, t]
            else:
                h = h_tilde + inj[:, t]
            outs.append(h)
        return torch.stack(outs, 1).view(N, d, H, W), h


# ----------------------------------------------------------------------------
# 门控 (M4): none | input (保持并行) | innov (完整新息门, 强制序贯)
# ----------------------------------------------------------------------------
class Gate(nn.Module):
    def __init__(self, d_state: int, mode: str = "input"):
        super().__init__()
        assert mode in ("none", "input", "innov")
        self.mode = mode
        if mode != "none":
            in_ch = 2 * d_state if mode == "input" else 3 * d_state
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, d_state, 1), nn.SiLU(),
                nn.Conv2d(d_state, d_state, 3, padding=1, groups=d_state),
                nn.SiLU(), nn.Conv2d(d_state, d_state, 1),
            )

    def forward(self, u: torch.Tensor, B: int, T: int,
                pred: torch.Tensor | None = None,
                prev: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "none":
            return u
        d = u.shape[1]
        u5 = u.view(B, T, d, *u.shape[-2:])
        first = (u5[:, :1] - prev.view(B, 1, d, *u.shape[-2:])
                 if prev is not None else torch.zeros_like(u5[:, :1]))
        du = torch.cat([first, u5[:, 1:] - u5[:, :-1]], dim=1).view_as(u)
        feats = [u, du] if self.mode == "input" else [u, du, pred]
        return torch.sigmoid(self.net(torch.cat(feats, dim=1))) * u


# ----------------------------------------------------------------------------
# 主模块
# ----------------------------------------------------------------------------
class SpecTransSSM(nn.Module):
    """
    yaml args: [d_state, seq_len, mode, use_adv, use_diff, use_decay,
                selective, gate, v_max, pad]
    c1 由 parse_model 注入 (见文末补丁); c1=None 时用 Lazy 卷积推断。
    """

    def __init__(self, c1: int | None = None, d_state: int = 32, seq_len: int = 8,
                 mode: str = "spectral", use_adv: bool = True, use_diff: bool = True,
                 use_decay: bool = True, selective: bool = True, gate: str = "input",
                 v_max: float = 4.0, pad: int = 8, bg_ratio: float = 0.5,
                 scan: str = "parallel"):
        super().__init__()
        assert mode in ("spectral", "local", "hybrid", "diag")
        if mode == "diag":
            mode, use_adv, use_diff = "spectral", False, False
        if gate == "innov":
            assert mode in ("local", "hybrid"), \
                "完整新息门依赖演化状态, 破坏扫描结合律: 仅 local/hybrid 序贯路径支持 " \
                "(spectral 并行请用 gate='input' 代理门 -- 这本身是消融轴 E6)"
            scan = "sequential"
        self.mode, self.T, self.d = mode, seq_len, d_state
        self._innov = gate == "innov"
        self.in_proj = (nn.Conv2d(c1, d_state, 1) if c1
                        else nn.LazyConv2d(d_state, 1))
        self.gate = Gate(d_state, "none" if self._innov else gate)
        if mode in ("spectral",):
            self.spec = SpectralTransport(d_state, v_max, selective, use_adv,
                                          use_diff, use_decay, pad, scan)
        elif mode == "local":
            self.coeff_g = SpectralTransport(d_state, v_max, selective, use_adv,
                                             use_diff, use_decay, pad, scan)
            self.local = LocalTransport(d_state, innov_gate=self._innov)
        else:  # hybrid
            self.d_bg = max(1, int(d_state * bg_ratio))
            self.d_tg = d_state - self.d_bg
            self.spec = SpectralTransport(self.d_bg, v_max, selective, use_adv,
                                          use_diff, use_decay, pad, scan)
            self.coeff_g = SpectralTransport(self.d_tg, v_max, selective, True,
                                             False, True, pad, scan)
            self.local = LocalTransport(self.d_tg, innov_gate=self._innov)
        self.out_proj = (nn.Conv2d(d_state, c1, 1) if c1
                         else nn.LazyConv2d(0, 1))  # 占位, forward 首次替换
        self._c1 = c1
        if c1:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        self.streaming = False
        self._state = None
        self._prev_u = None

    # ---- 流式 API (Protocol-S) ----
    def reset_state(self):
        self._state = None
        self._prev_u = None

    def _ensure_out(self, x):
        if self._c1 is None and isinstance(self.out_proj, nn.LazyConv2d):
            self.out_proj = nn.Conv2d(self.d, x.shape[1], 1).to(x.device)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            self._c1 = x.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        cached_T = ClipToBatch._KEY_BYPASS.get("T")
        if self.streaming:
            T = 1
        elif cached_T is not None and N % cached_T == 0:
            T = cached_T                                  # 权威: 来自 ClipToBatch
        elif N % self.T == 0:
            T = self.T                                    # 回退: yaml seq_len
        else:
            T = 1
            if self.training and not getattr(self, "_warned_T", False):
                print(f"[SpecTransSSM] WARNING: N={N} 既不能被缓存 T={cached_T} 也不能被 "
                      f"seq_len={self.T} 整除, 回退 T=1 (本层无时序递归!). "
                      f"检查 ClipToBatch 是否在 backbone 入口, 及通道排布契约。")
                self._warned_T = True
        B = N // T
        self._ensure_out(x)
        u = self.in_proj(x)
        prev = self._prev_u if self.streaming else None
        inj = self.gate(u, B, T, prev=prev)
        if self.streaming:
            self._prev_u = u.view(B, T, *u.shape[1:])[:, -1].detach()  # innov 门的 pred 项在序贯路径内闭合, 此处简化为 input 分支
        if self.mode == "spectral":
            h0 = self._state if self.streaming else None
            h, h_last, _ = self.spec(inj, u, B, T, h0)
            if self.streaming:
                self._state = h_last.detach()
        elif self.mode == "local":
            vx, vy, *_ = self.coeff_g.coeff(u, B, T)   # 只取全局 v, 不走谱状态
            vg = (vx, vy)
            h0 = self._state if self.streaming else None
            h, h_last = self.local(inj, u, B, T, vg, h0)
            if self.streaming:
                self._state = h_last.detach()
        else:  # hybrid
            u_bg, u_tg = u[:, :self.d_bg], u[:, self.d_bg:]
            i_bg, i_tg = inj[:, :self.d_bg], inj[:, self.d_bg:]
            s_bg = self._state[0] if (self.streaming and self._state) else None
            s_tg = self._state[1] if (self.streaming and self._state) else None
            h_bg, hb_last, _ = self.spec(i_bg, u_bg, B, T, s_bg)
            vx, vy, *_ = self.coeff_g.coeff(u_tg, B, T)
            vg = (vx, vy)
            h_tg, ht_last = self.local(i_tg, u_tg, B, T, vg, s_tg)
            h = torch.cat([h_bg.to(h_tg.dtype), h_tg], dim=1)
            if self.streaming:
                self._state = (hb_last.detach(), ht_last.detach())
        return x + self.out_proj(h.to(x.dtype))


# ----------------------------------------------------------------------------
# M1: SPD-Conv (space-to-depth 无损下采样), 供强化单帧锚点 yaml 使用
# ----------------------------------------------------------------------------
class SPDConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(4 * c1, c2, k, 1, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(F.pixel_unshuffle(x, 2))))


# ----------------------------------------------------------------------------
# Ultralytics 集成
# ----------------------------------------------------------------------------
def register_trace_modules():
    """
    将模块注入 ultralytics 命名空间。之后仍需在 ultralytics/nn/tasks.py 的
    parse_model 中加入通道推断分支 (与你们 phys_ss2d 的接法一致):

        elif m is SpecTransSSM:
            c1 = ch[f]; c2 = c1
            args = [c1, *args]
        elif m is SPDConv:
            c1, c2 = ch[f], make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
    """
    import ultralytics.nn.tasks as tasks
    for cls in (SpecTransSSM, SPDConv, ClipToBatch, BatchToClip):
        setattr(tasks, cls.__name__, cls)


# ============================================================================
# 视频 clip 的解包/打包 (匹配 "5 灰度帧 + 3 通道关键帧RGB = 8ch 通道堆叠" 排布)
# ----------------------------------------------------------------------------
# 数据契约 (来自用户数据集): 输入 (B, 8, H, W)
#   通道 0..4 : 5 个连续灰度帧 (时序序列, index 0=最早, 4=当前/关键帧)
#   通道 5..7 : 当前帧复制成的 RGB (关键帧外观旁路, 不参与时序递归)
#
# ClipToBatch  放在 backbone 第 0 层之前: 把 5 帧沿 batch 维展开 → (B*5,1,H,W),
#              并把关键帧 RGB 旁路缓存到全局, 供 BatchToClip 取回。
#              这样 batch 维携带 T, DDP/BN/增强全部无感, seq_len 恒等匹配, 无整除问题。
# BatchToClip  放在 neck 汇入检测头之前的每条尺度支路: 把 (B*5,C,h,w) 折回,
#              取末帧 (当前帧) 状态 (B,C,h,w) 作为检测特征, 完成时序→单帧的收束。
# ============================================================================
class ClipToBatch(nn.Module):
    """(B, n_gray+key_rgb, H, W) → (B*T, stem_ch, H, W); 关键帧旁路进 buffer。
    yaml args: [n_gray, key_rgb, stem_ch]  例: [5, 3, 1] 表示 5 灰度帧 + 3 RGB。
    stem_ch=1: 每帧单通道喂 backbone; =3: 每帧复制成3通道 (兼容 ImageNet 预训练 stem)。
    """
    _KEY_BYPASS = {}  # id(model)->tensor; 用类属性跨层传递, 避免改 tasks.py 数据流

    def __init__(self, n_gray: int = 5, key_rgb: int = 3, stem_ch: int = 1):
        super().__init__()
        self.n_gray, self.key_rgb, self.stem_ch = n_gray, key_rgb, stem_ch

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.n_gray + self.key_rgb, \
            f"ClipToBatch 期望 {self.n_gray}+{self.key_rgb} 通道, 实得 {C}"
        gray = x[:, :self.n_gray]                       # (B, T, H, W)
        ClipToBatch._KEY_BYPASS["last"] = x[:, self.n_gray:].contiguous()
        ClipToBatch._KEY_BYPASS["B"] = B
        ClipToBatch._KEY_BYPASS["T"] = self.n_gray
        frames = gray.reshape(B * self.n_gray, 1, H, W)  # index = b*T+t ✓
        if self.stem_ch > 1:
            frames = frames.expand(-1, self.stem_ch, -1, -1)
        return frames


class BatchToClip(nn.Module):
    """(B*T, C, h, w) → (B, C[+key], h, w)。fuse='last' 取当前帧; 'mean' 取时序均值。
    add_key=True 时把关键帧 RGB 旁路下采样到本尺度并 concat (需 out_ch 对齐, 见 yaml)。
    yaml args: [fuse, add_key]  例: ['last', False]
    """
    def __init__(self, fuse: str = "last", add_key: bool = False):
        super().__init__()
        assert fuse in ("last", "mean")
        self.fuse, self.add_key = fuse, add_key

    def forward(self, x):
        B = ClipToBatch._KEY_BYPASS.get("B")
        T = ClipToBatch._KEY_BYPASS.get("T")
        assert B is not None, "BatchToClip 未找到 ClipToBatch 缓存的 (B,T); 检查 yaml 顺序"
        C, h, w = x.shape[1:]
        x = x.view(B, T, C, h, w)
        out = x[:, -1] if self.fuse == "last" else x.mean(1)
        if self.add_key:
            key = ClipToBatch._KEY_BYPASS["last"]
            key = F.interpolate(key, size=(h, w), mode="bilinear",
                                align_corners=False)
            out = torch.cat([out, key], dim=1)
        return out


# ============================================================================
# ultralytics/nn/tasks.py :: parse_model 需要加入以下分支 (放在现有 elif 链中):
#
#     elif m is ClipToBatch:
#         c2 = args[2]                       # stem_ch, 后续首个 Conv 的 in_ch
#         # args 原样传入: [n_gray, key_rgb, stem_ch]
#     elif m is BatchToClip:
#         c2 = ch[f] + (3 if len(args) > 1 and args[1] else 0)
#         # args 原样传入: [fuse, add_key]
#     elif m is SpecTransSSM:
#         c1 = ch[f]; c2 = c1                 # 残差, 通道不变
#         args = [c1, *args]
#     elif m is SPDConv:
#         c1 = ch[f]
#         c2 = make_divisible(min(args[0], max_channels) * width, 8)
#         args = [c1, c2, *args[1:]]
#
# 数据侧: 训练器需设 channels=8 (DAUB.yaml 里 ch: 8 或等价), frame_num=5。
# 关键校验: 训练启动后, 日志里若再出现 "回退 T=1" 的 WARNING, 说明契约仍未接通,
#           必须停下来查 —— 正确接通时该 WARNING 永不出现。
# ============================================================================
