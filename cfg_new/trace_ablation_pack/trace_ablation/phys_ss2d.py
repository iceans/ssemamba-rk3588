# -*- coding: utf-8 -*-
"""
PhysSS2D — VMamba SS2D(帧内空间选择性扫描) × 物理对流–扩散递归(帧间状态转移)
================================================================================
设计分工（对应论文故事线）:
    * 帧内: VMamba 的 SS2D 做 2D 选择性扫描 —— 提供空间上下文与选择性系数的"量测编码";
    * 帧间: 状态转移方程被替换为空间离散化的对流–扩散–衰减方程 —— 这正是"用物理传导
      方程替换 SSM 状态转移"的落点, 递归只沿 T 维进行(T 小, 无需 parallel scan):

          dH/dt = -(v·∇)H + α∇²H − λH + b ⊙ U(z_t),      z_t = SS2D(x_t)

      v/α/λ/b 逐像素由当前帧预测("选择性 = 物理系数的输入依赖")。

数值格式(每帧, 算子分裂):
    1) 半拉格朗日对流  H ← warp(H, −v)      grid_sample 反向回溯, 无条件稳定;
    2) 显式扩散 K 子步  H ← H + α∇²H         α = 0.25·σ(·) 逐子步满足 2D CFL;
    3) 精确衰减+注入    H ← e^{−λ}H + (1−e^{−λ})·b⊙U(z_t)   与 Mamba ZOH 同构。

消融矩阵(全部可从 yaml 直接开关):
    use_ss2d=False                        → 退化为纯物理时序块(上一版 PhysSSM)
    use_advection=False, use_diffusion=False → 逐像素时序 Mamba(对角 A) —— 关键对照
    use_advection=False                   → 纯扩散(vHeat 式空间聚合 + 时间递归)
    use_diffusion=False                   → 纯输运

依赖:
    * 若安装了 mamba_ssm(或 VMamba 的 selective_scan CUDA 核), 自动使用 CUDA 扫描;
    * 否则退回"分块并行扫描"的纯 PyTorch 实现(数值稳定: 块内衰减矩阵恒 ≤ 1)。
      纯 PyTorch 路径能跑通训练/调试, 但速度和显存都差, 正式实验请装 CUDA 核:
          pip install mamba-ssm  (或编译 VMamba 仓库中的 kernels/selective_scan)

================================================================================
Ultralytics 集成(三步):
  1) 本文件放到  ultralytics/nn/modules/phys_ss2d.py
  2) 在 ultralytics/nn/tasks.py 顶部加:
         from ultralytics.nn.modules.phys_ss2d import PhysSS2D, TemporalUnfold, TemporalSelect
     并在 parse_model() 的 c2 判断链(一串 if m in {...} / elif)中加入:
         elif m is TemporalUnfold:
             c2 = args[2] if len(args) > 2 else 1   # 输出通道 = out_channels 参数
         elif m is KeyFrame:
             c2 = args[1] if len(args) > 1 else 3   # 关键帧通道数
         elif m is TemporalSelect:
             c2 = ch[f]
         elif m is PhysSS2D:
             c2 = ch[f]
             args = [ch[f], *args]       # 自动注入输入通道数 c1
  3) 使用 yolov8-physs2d.yaml(单尺度, 推荐先跑) 或 yolov8-trace.yaml(完整框架)。
     ★ 输入格式约定(与数据集实现对齐): 数据张量为 (B, T+K, H, W) ——
       前 T 个通道是时序灰度帧堆叠, 末 K 个通道是关键帧(=最后一帧)的 K 通道表示
       (典型: T=5, K=3, 总通道 8)。三处必须同步:
         模型 yaml 的 ch: = T+K;  TemporalUnfold 参数 [T, K, out];
         PhysSS2D/TraceSS2D 的 nframes = T;  数据集 yaml 的 channels: = T+K。

训练/推理注意:
  * keep='last' 时检测头 batch=B, 损失按标准 YOLO 走, 标签取每个 clip 的"末帧"标注;
    keep='all' 时预测是 B·T 帧, 需把标签的 batch_idx 映射为 b*T + t(逐帧监督)。
  * AMP 下 grid_sample 与 selective scan 内部强制 fp32, 已在模块内处理。
  * 在线跟踪: 用 forward_stream(x_t, state) 逐帧滚动, 显存/延迟为常数;
    state 中的速度场 v 可读出作为关联阶段的运动先验。
"""

import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. selective scan: CUDA 核(可选) + 纯 PyTorch 分块并行扫描(兜底)
# ----------------------------------------------------------------------------
try:  # mamba_ssm 官方核, 与 VMamba 调用方式兼容(B/C 带方向组维)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _scan_cuda
    _HAS_CUDA_SCAN = True
except Exception:
    _scan_cuda, _HAS_CUDA_SCAN = None, False

_WARNED = False


def selective_scan_torch(u, delta, A, Bs, Cs, Ds=None, delta_bias=None,
                         delta_softplus=True, chunk=None):
    """纯 PyTorch 分块并行扫描。
    u/delta: (b, k*d, l);  A: (k*d, n);  Bs/Cs: (b, k, n, l);  Ds/delta_bias: (k*d,)
    数值稳定性: 块内衰减矩阵 M[t,j] = exp(S_t − S_j), j ≤ t 时恒 ≤ 1(dA ≤ 0), 无溢出。
    复杂度换取并行度: 每块构造 (c×c) 成对衰减矩阵, chunk 默认 32(可用环境变量
    PHYS_SCAN_CHUNK 调节: 大 → 快但费显存)。仅作跑通/调试用, 训练请装 CUDA 核。
    """
    chunk = chunk or int(os.environ.get("PHYS_SCAN_CHUNK", 32))
    b, kd, l = u.shape
    k, n = Bs.shape[1], Bs.shape[2]
    d = kd // k

    u32, dt = u.float(), delta.float()
    if delta_bias is not None:
        dt = dt + delta_bias.float().view(1, kd, 1)
    if delta_softplus:
        dt = F.softplus(dt)

    ug = u32.view(b, k, d, l)
    dtg = dt.view(b, k, d, l)
    Ag = A.float().view(k, d, n)
    dA = torch.einsum("bkdl,kdn->bkdln", dtg, Ag)                 # ≤ 0 (A<0, dt≥0)
    dBu = torch.einsum("bkdl,bknl,bkdl->bkdln", dtg, Bs.float(), ug)

    h = u32.new_zeros(b, k, d, n)
    ys = []
    for s in range(0, l, chunk):
        e = min(s + chunk, l)
        c = e - s
        S = dA[..., s:e, :].cumsum(dim=3)                          # (b,k,d,c,n)
        diff = S.unsqueeze(4) - S.unsqueeze(3)                     # S_t − S_j
        tril = torch.ones(c, c, dtype=torch.bool, device=u.device).tril()
        M = diff.masked_fill(~tril.view(1, 1, 1, c, c, 1), float("-inf")).exp()
        hc = torch.einsum("bkdtjn,bkdjn->bkdtn", M, dBu[..., s:e, :])
        hc = hc + S.exp() * h.unsqueeze(3)                         # 携带块首状态
        ys.append(torch.einsum("bkdtn,bknt->bkdt", hc, Cs[..., s:e].float()))
        h = hc[:, :, :, -1]
    y = torch.cat(ys, dim=3).reshape(b, kd, l)
    if Ds is not None:
        y = y + u32 * Ds.float().view(1, kd, 1)
    return y.to(u.dtype)


def selective_scan(u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus=True):
    global _WARNED
    if _HAS_CUDA_SCAN and u.is_cuda:
        return _scan_cuda(u, delta, A, Bs, Cs, Ds, None, delta_bias,
                          delta_softplus, False)
    if not _WARNED:
        warnings.warn("[PhysSS2D] 未检测到 CUDA selective scan, 使用纯 PyTorch 兜底"
                      "(慢且费显存, 仅供跑通; 正式训练请安装 mamba-ssm)。")
        _WARNED = True
    return selective_scan_torch(u, delta, A, Bs, Cs, Ds, delta_bias, delta_softplus)


# ----------------------------------------------------------------------------
# 1. SS2D — VMamba 式 2D 选择性扫描(4 方向交叉扫描)
# ----------------------------------------------------------------------------
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, expand=1.0, d_conv=3, dt_rank=None,
                 dt_min=0.001, dt_max=0.1, dt_scale=1.0, dt_init_floor=1e-4,
                 bias=False, conv_bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        K = self.K = 4

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv,
                                padding=(d_conv - 1) // 2, groups=self.d_inner,
                                bias=conv_bias)
        self.act = nn.SiLU()

        # 每方向 (Δ, B, C) 投影, 合并成单个参数张量
        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False).weight
            for _ in range(K)]))                                   # (K, R+2N, D)
        dtp = [self._dt_init(self.dt_rank, self.d_inner, dt_scale,
                             dt_min, dt_max, dt_init_floor) for _ in range(K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dtp]))  # (K,D,R)
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dtp]))      # (K,D)

        self.A_logs = nn.Parameter(self._A_log_init(d_state, self.d_inner, K))     # (K*D,N)
        self.Ds = nn.Parameter(torch.ones(K * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_min, dt_max, floor):
        p = nn.Linear(dt_rank, d_inner, bias=True)
        std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(p.weight, -std, std)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=floor)
        with torch.no_grad():
            p.bias.copy_(dt + torch.log(-torch.expm1(-dt)))        # softplus 逆
        return p

    @staticmethod
    def _A_log_init(d_state, d_inner, copies):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        return torch.log(A).unsqueeze(0).repeat(copies, 1, 1).flatten(0, 1)

    def forward_core(self, x):                                     # x: (B, D, H, W)
        B, D, H, W = x.shape
        L, K = H * W, self.K
        # 4 方向: 行主序 / 列主序 + 各自反向
        x_hwwh = torch.stack([
            x.reshape(B, D, L),
            x.transpose(2, 3).contiguous().reshape(B, D, L)], dim=1)
        xs = torch.cat([x_hwwh, x_hwwh.flip(-1)], dim=1)           # (B, 4, D, L)

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        out_y = selective_scan(
            xs.reshape(B, K * D, L).float(),
            dts.contiguous().reshape(B, K * D, L).float(),
            -torch.exp(self.A_logs.float()),
            Bs.float(), Cs.float(), self.Ds.float(),
            self.dt_projs_bias.reshape(-1).float(),
        ).view(B, K, D, L)

        inv = out_y[:, 2:4].flip(-1)                               # 反向两路翻回
        y = (out_y[:, 0] + inv[:, 0]
             + out_y[:, 1].view(B, D, W, H).transpose(2, 3).reshape(B, D, L)
             + inv[:, 1].view(B, D, W, H).transpose(2, 3).reshape(B, D, L))
        return y.transpose(1, 2).reshape(B, H, W, D)

    def forward(self, x):                                          # x: (B, H, W, C)
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y = self.out_norm(self.forward_core(x).to(z.dtype))
        return self.out_proj(y * F.silu(z))


# ----------------------------------------------------------------------------
# 2. PhysSS2D — ultralytics 模块: SS2D(帧内) + 物理输运递归(帧间)
# ----------------------------------------------------------------------------
class PhysSS2D(nn.Module):
    """输入 (B·T, C, H, W)(帧折叠进 batch), keep='all' 输出同形状, 'last' 输出 (B, C, H, W)。"""

    def __init__(self, c1, nframes=5, d_state=16, expand=1.0, groups=4,
                 v_max=3.0, keep="last", diff_substeps=2,
                 use_ss2d=True, use_advection=True, use_diffusion=True):
        super().__init__()
        assert keep in ("all", "last") and c1 % groups == 0
        self.T, self.g, self.n = nframes, groups, c1
        self.v_max, self.k_diff, self.alpha_cfl = v_max, diff_substeps, 0.25
        self.keep = keep
        self.use_ss2d, self.use_adv, self.use_dif = use_ss2d, use_advection, use_diffusion

        if use_ss2d:                                   # 帧内: VSS 残差 (LN → SS2D)
            self.ln = nn.LayerNorm(c1)
            self.ss2d = SS2D(c1, d_state=d_state, expand=expand)

        G, N = groups, c1                              # 帧间: 物理系数头 v/α/λ/b
        self.coef = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1), nn.SiLU(),
            nn.Conv2d(c1, 2 * G + G + N + N, 1))
        self.U = nn.Conv2d(c1, N, 1)                   # 源项(量测)投影
        self.Cgate = nn.Conv2d(c1, N, 1)               # 输出门(对应 Mamba 的 C)
        self.norm = nn.GroupNorm(1, N)
        self.out = nn.Conv2d(N, c1, 1)

        final = self.coef[-1]                          # 初始化: 恒等 + 轻微 EMA
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[2 * G:3 * G] = -2.0             # α ≈ 0.03
            final.bias[3 * G:3 * G + N] = -2.2         # e^{−λ} ≈ 0.9, 记忆 ~10 帧
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("lap", lap.view(1, 1, 3, 3).repeat(N, 1, 1, 1))
        self._grid = {}

    # ---- 帧间递归的三步数值格式 -------------------------------------------
    def _base_grid(self, h, w, dev):
        dev = torch.device(dev)
        key = (h, w, str(dev))
        cached = self._grid.get(key)
        # 关键修复: 缓存是普通 dict, 不随 model.to(device) 移动; 若整模型被
        # pickle 保存/加载(ultralytics 默认存整份模型对象)后再搬到别的设备,
        # key 会命中但张量本体还停在旧 device 上——必须用真实 tensor.device
        # 校验, 而不是只信任 key 里记录的字符串。
        if cached is None or cached.device != dev:
            yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                    torch.arange(w, device=dev, dtype=torch.float32),
                                    indexing="ij")
            cached = torch.stack((xx, yy), -1)                    # (h, w, 2), (x, y)
            self._grid[key] = cached
        return cached

    def __getstate__(self):
        # 保存(pickle)时丢弃网格缓存, 避免旧 device 上的张量被存进 checkpoint;
        # 加载后首次前向会在正确的 device 上重新生成, 见 _base_grid 的校验逻辑。
        state = self.__dict__.copy()
        state["_grid"] = {}
        return state

    def _advect(self, H, v):                           # 半拉格朗日, 内部 fp32
        B, N, h, w = H.shape
        G = self.g
        Hg = H.float().view(B * G, N // G, h, w)
        vg = v.float().view(B * G, 2, h, w).permute(0, 2, 3, 1)
        src = self._base_grid(h, w, H.device) - vg
        gx = src[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
        gy = src[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        out = F.grid_sample(Hg, torch.stack((gx, gy), -1), mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.view(B, N, h, w).to(H.dtype)

    def step(self, z_t, state=None):
        """单帧递归。z_t: (B, C, h, w) —— 已过帧内 SS2D 的特征。"""
        B, C, h, w = z_t.shape
        G, N = self.g, self.n
        if state is None:
            state = z_t.new_zeros(B, N, h, w)
        coef = self.coef(z_t)
        o = 0
        v = self.v_max * torch.tanh(coef[:, o:o + 2 * G]); o += 2 * G
        alpha = self.alpha_cfl * torch.sigmoid(coef[:, o:o + G]); o += G
        lam = F.softplus(coef[:, o:o + N]); o += N
        b_gate = torch.sigmoid(coef[:, o:o + N])

        H = state
        if self.use_adv:                               # 1) 对流(输运)
            H = self._advect(H, v)
        if self.use_dif:                               # 2) 扩散(CFL 安全显式子步)
            a = alpha.repeat_interleave(N // G, dim=1)
            for _ in range(self.k_diff):
                H = H + a * F.conv2d(H, self.lap, padding=1, groups=N)
        decay = torch.exp(-lam)                        # 3) 衰减 + 源注入(ZOH)
        H = decay * H + (1.0 - decay) * (b_gate * self.U(z_t))

        y = self.norm(F.silu(self.Cgate(z_t)) * H)
        return z_t + self.out(y), H

    # ---- 前向 ---------------------------------------------------------------
    def _spatial(self, x):                             # 帧内 VSS 残差
        if not self.use_ss2d:
            return x
        return x + self.ss2d(self.ln(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

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
        assert BT % T == 0, f"batch({BT}) 不能被 nframes({T}) 整除, 检查 yaml 的 ch/nframes 与数据集 T"
        z = self._spatial(x).view(BT // T, T, C, H, W)
        state, outs = None, []
        for t in range(T):
            o, state = self.step(z[:, t], state)
            outs.append(o)
        out = torch.stack(outs, 1)
        return out[:, -1] if self.keep == "last" else out.reshape(BT, C, H, W)

    @torch.no_grad()
    def forward_stream(self, x_t, state=None):
        """在线推理: 单帧 (B, C, H, W) + 外部持有的 state, 常数显存逐帧滚动。"""
        return self.step(self._spatial(x_t), state)


# ----------------------------------------------------------------------------
# 3. 时序数据流辅助模块
# ----------------------------------------------------------------------------
class TemporalUnfold(nn.Module):
    """把时序堆叠输入拆成逐帧批次, 支持三种输入格式:

      (B, T, C, H, W)   → (B·T, C, H, W)                    [5D 直接折叠]
      (B, T, H, W)      → (B·T, out_channels, H, W)         [纯帧堆叠, nframes=0 或 =T]
      (B, T+K, H, W)    → (B·T, out_channels, H, W)         [★ 帧堆叠 + 关键帧格式]

    ★ 关键帧格式(nframes=T, key_channels=K>0): 前 T 个通道是灰度帧序列,
      末 K 个通道是关键帧(=最后一帧)的 K 通道表示(典型 T=5, K=3, 共 8 通道)。
      - out_channels==K 时: 各帧灰度复制到 K 通道, 而"最后一帧"槽位直接放入
        关键帧块——语义与"末帧=关键帧"完全一致, 且 conv1 变为 K(=3) 通道,
        可加载 COCO 预训练权重;
      - out_channels==1 时: 只取 T 个灰度帧, 关键帧块被丢弃(其信息与末帧冗余)。
    """

    def __init__(self, nframes=0, key_channels=0, out_channels=1):
        super().__init__()
        self.T, self.K, self.C = nframes, key_channels, out_channels

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        B, Ctot, H, W = x.shape
        T = self.T if self.T > 0 else Ctot - self.K
        assert Ctot == T + self.K, \
            f"输入通道 {Ctot} ≠ nframes({T}) + key_channels({self.K}), 检查数据集与 yaml 的 ch:"
        f = x[:, :T].unsqueeze(2)                        # (B, T, 1, H, W)
        if self.C > 1:
            f = f.expand(B, T, self.C, H, W)             # 灰度复制到 out_channels
        if self.K > 0 and self.C == self.K:
            f = torch.cat([f[:, :-1], x[:, T:].unsqueeze(1)], dim=1)  # 末帧槽位=关键帧块
        return f.reshape(B * T, self.C, H, W)


class KeyFrame(nn.Module):
    """(B, T+K, H, W) → (B, K, H, W): 抽出末尾 K 个通道(关键帧), 供独立的
    关键帧支路使用(如需在某处与时序特征融合, 用 from: 引用本层输出)。"""

    def __init__(self, nframes=5, key_channels=3):
        super().__init__()
        self.T, self.K = nframes, key_channels

    def forward(self, x):
        assert x.shape[1] == self.T + self.K, \
            f"输入通道 {x.shape[1]} ≠ nframes({self.T}) + key_channels({self.K})"
        return x[:, self.T:]


class TemporalSelect(nn.Module):
    """(B·T, C, H, W) → (B, C, H, W), 取第 index 帧(默认末帧)。"""
    def __init__(self, nframes, index=-1):
        super().__init__()
        self.T, self.i = nframes, index

    def forward(self, x):
        BT, C, H, W = x.shape
        return x.view(BT // self.T, self.T, C, H, W)[:, self.i]


# ----------------------------------------------------------------------------
# 4. 自检
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # (a) 分块并行扫描 vs 朴素串行递归: 数学等价性
    def naive_scan(u, delta, A, Bs, Cs, Ds, bias):
        b, kd, l = u.shape
        k, n = Bs.shape[1], Bs.shape[2]
        d = kd // k
        dt = F.softplus(delta + bias.view(1, kd, 1))
        ug, dtg, Ag = u.view(b, k, d, l), dt.view(b, k, d, l), A.view(k, d, n)
        h, ys = u.new_zeros(b, k, d, n), []
        for t in range(l):
            h = (torch.exp(dtg[..., t].unsqueeze(-1) * Ag) * h
                 + dtg[..., t].unsqueeze(-1) * Bs[..., t].unsqueeze(2) * ug[..., t].unsqueeze(-1))
            ys.append(torch.einsum("bkdn,bkn->bkd", h, Cs[..., t]))
        return torch.stack(ys, -1).reshape(b, kd, l) + u * Ds.view(1, kd, 1)

    b, k, d, n, l = 2, 4, 3, 8, 53
    u = torch.randn(b, k * d, l)
    delta = torch.randn(b, k * d, l)
    A = -torch.rand(k * d, n) - 0.5
    Bs, Cs = torch.randn(b, k, n, l), torch.randn(b, k, n, l)
    Ds, bias = torch.randn(k * d), torch.randn(k * d)
    y1 = selective_scan_torch(u, delta, A, Bs, Cs, Ds, bias, chunk=8)
    y2 = naive_scan(u, delta, A, Bs, Cs, Ds, bias)
    print("chunked scan == naive scan:", torch.allclose(y1, y2, atol=1e-4))

    # (b) PhysSS2D 前向/反向/两种 keep 模式
    # 注: 兜底扫描 + 自动求导会缓存每块的 (c×c) 衰减矩阵, 自检用小尺寸/小分块;
    #     真实训练在 GPU + CUDA 核下无此开销。
    os.environ["PHYS_SCAN_CHUNK"] = "16"
    B, T, C, H, W = 2, 4, 16, 24, 24
    x = torch.randn(B * T, C, H, W, requires_grad=True)
    m_all = PhysSS2D(C, nframes=T, groups=4, keep="all")
    m_last = PhysSS2D(C, nframes=T, groups=4, keep="last")
    ya, yl = m_all(x), m_last(x)
    print("keep=all :", tuple(x.shape), "->", tuple(ya.shape))
    print("keep=last:", tuple(x.shape), "->", tuple(yl.shape))
    yl.mean().backward()
    print("backward : ok, grad norm =", float(x.grad.norm()))
    print(f"params(C={C}):", sum(p.numel() for p in m_last.parameters()))

    # (c) 流式推理与 clip 前向一致性(keep='all' 下逐帧对比)
    with torch.no_grad():
        state, outs = None, []
        for t in range(T):
            o, state = m_all.forward_stream(x.view(B, T, C, H, W)[:, t], state)
            outs.append(o)
        ys = torch.stack(outs, 1).reshape(B * T, C, H, W)
        print("streaming == clip forward:", torch.allclose(ya, ys, atol=1e-4))

    # (e) 8 通道输入格式(5 灰度帧 + 3 通道关键帧)
    tu = TemporalUnfold(5, 3, 3)
    xin = torch.randn(2, 8, 32, 32)
    fo = tu(xin)
    f5 = fo.view(2, 5, 3, 32, 32)
    assert fo.shape == (10, 3, 32, 32)
    assert torch.equal(f5[:, -1], xin[:, 5:])            # 末帧槽位 = 关键帧块
    assert torch.equal(f5[:, 0, 0], f5[:, 0, 1])         # 其余帧为灰度复制
    print("TemporalUnfold(5,3,3):", tuple(xin.shape), "->", tuple(fo.shape),
          "| key-frame slot ok")
    print("TemporalUnfold(5,3,1):", tuple(TemporalUnfold(5, 3, 1)(xin).shape))
    print("KeyFrame(5,3)       :", tuple(KeyFrame(5, 3)(xin).shape))

    # (d) 消融开关形状检查: 对角 A(逐像素时序 Mamba) / 关 SS2D
    m_diag = PhysSS2D(C, nframes=T, use_advection=False, use_diffusion=False)
    m_noss = PhysSS2D(C, nframes=T, use_ss2d=False)
    print("diag-A   :", tuple(m_diag(x).shape), "| no-ss2d :", tuple(m_noss(x).shape))
