# from .common_utils_tgsmamba import *
import math
from functools import partial
from typing import Optional, Callable, Any
from timm.models.layers import DropPath, trunc_normal_
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from ultralytics.nn.modules.conv import Conv
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False
WITH_SELECTIVESCAN_OFLEX = True
try:
    import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
WITH_SELECTIVESCAN_CORE = True
try:
    import selective_scan_cuda_core
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
import warnings

WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")


class SS2Dv2(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            inchannel,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.inchannel = inchannel
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type ==============================
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
                             bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            x_dbl = self.x_proj(xs.view(B, -1, L))
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
# class SS2D(nn.Module):
#     def __init__(
#             self,
#             # basic dims ===========
#             d_model=96,
#             d_state=16,
#             ssm_ratio=2.0,
#             ssm_rank_ratio=2.0,
#             dt_rank="auto",
#             act_layer=nn.SiLU,
#             # dwconv ===============
#             d_conv=1,  # < 2 means no conv
#             conv_bias=True,
#             # ======================
#             dropout=0.0,
#             bias=False,
#             # ======================
#             forward_type="v2",
#             channel_first=False,#todo
#             **kwargs,
#     ):
#         """
#         ssm_rank_ratio would be used in the future...
#         """
#         factory_kwargs = {"device": None, "dtype": None}
#         super().__init__()
#         d_expand = int(ssm_ratio * d_model)
#         d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
#         self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
#         self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
#         self.d_conv = d_conv
#         self.K = 1
#
#         # tags for forward_type ==============================
#         def checkpostfix(tag, value):
#             ret = value[-len(tag):] == tag
#             if ret:
#                 value = value[:-len(tag)]
#             return ret, value
#
#         self.disable_force32, forward_type = checkpostfix("no32", forward_type)
#         self.disable_z, forward_type = checkpostfix("noz", forward_type)
#         self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)
#
#         self.out_norm = nn.LayerNorm(d_inner)
#
#         # forward_type debug =======================================
#         FORWARD_TYPES = dict(
#             v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
#         )
#         self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))
#
#         # in proj =======================================
#         d_proj = d_expand if self.disable_z else (d_expand * 2)
#         # self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
#         self.in_proj = Linear(self.d_model, d_proj, bias=bias, channel_first=channel_first)
#         self.act: nn.Module = act_layer
#         # self.act: nn.Module = nn.GELU()
#
#         # conv =======================================
#         if self.d_conv > 1:
#             self.conv2d = nn.Conv2d(
#                 in_channels=d_expand,
#                 out_channels=d_expand,
#                 groups=d_expand,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#
#         # rank ratio =====================================
#         self.ssm_low_rank = False
#         if d_inner < d_expand:
#             self.ssm_low_rank = True
#             self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
#             self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)
#
#         # x proj ============================
#         self.x_proj = [
#             nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
#                       **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)#
#         del self.x_proj
#
#         # out proj =======================================
#         self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#
#         # simple init dt_projs, A_logs, Ds
#         self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
#         self.A_logs = nn.Parameter(
#             torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
#         self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))
#
#         #tod o BC
#
#         # self.conv_num = d_inner+ self.d_state * 2
#         # self.my_conv = BasicRFB(d_inner,self.conv_num)#
#
#         # self.my_conv = BasicConv(d_inner, self.conv_num, kernel_size=3, stride=1,padding=1)
#         # self.my_conv = nn.Conv2d(d_inner, self.conv_num, kernel_size=3, stride=1, padding=1)
#         # ##TO DO
#         # self.conv_norm = nn.BatchNorm2d(d_inner+ self.d_state * 2, eps=1e-5, momentum=0.01, affine=True)#to do BC
#         # # self.conv_norm = nn.BatchNorm2d(self.d_state+d_inner , eps=1e-5, momentum=0.01, affine=True)#to do BC
#         # # self.conv_norm = nn.BatchNorm2d(d_inner ,eps=1e-5, momentum=0.01, affine=True)
#         # #self.dt_norm_map_weight = nn.Parameter(torch.randn((self.K, d_inner, d_inner)))#引用单独注释
#         # # self.my_conv=nn.Conv2d(d_inner, d_inner, kernel_size=3, stride=1, bias=bias, **factory_kwargs)
#
#         #tod o conv and linear
#
#         self.conv_num = d_inner
#         # self.my_conv = BasicRFB(d_inner, self.conv_num)  #
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#
#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         # dt_proj.bias._no_reinit = True
#
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 0:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=-1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 0:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D
#
#     def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
#                        cross_selective_scan=cross_selective_scan, force_fp32=None):
#         force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
#         if not channel_first:
#             x = x.permute(0, 3, 1, 2).contiguous()
#         if self.ssm_low_rank:
#             x = self.in_rank(x)
#         x = cross_selective_scan(
#             x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
#             self.A_logs, self.Ds,
#             out_norm=getattr(self, "out_norm", None),
#             out_norm_shape=getattr(self, "out_norm_shape", "v0"),
#             delta_softplus=True, force_fp32=force_fp32,
#             SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
#             # my_conv = self.my_conv,
#             conv_num =self.conv_num,
#             # ##TODO
#             # conv_norm = self.conv_norm,
#             # # dt_norm_map_weight = self.dt_norm_map_weight#引用单独注释
#         )
#         if self.ssm_low_rank:
#             x = self.out_rank(x)
#         return x
#
#     def forward(self, x: torch.Tensor, **kwargs):
#         x = self.in_proj(x)
#         if not self.disable_z:
#             x, z = x.chunk(2, dim=1)  # (b, d, h, w)
#             if not self.disable_z_act:
#                 z1 = self.act(z)
#         if self.d_conv > 0:
#             x = self.conv2d(x)  # (b, d, h, w)
#         x = self.act(x)
#         y = self.forward_core(x, channel_first=(self.d_conv > 1))
#         y = y.permute(0, 3, 1, 2).contiguous()
#         if not self.disable_z:
#             y = y * z1
#         out = self.dropout(self.out_proj(y))
#         return out
@triton.jit
def triton_cross_scan_flex(
        x: tl.tensor,  # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        y: tl.tensor,  # (B, 4, C, H, W) | (B, H, W, 4, C)
        x_layout: tl.constexpr,
        y_layout: tl.constexpr,
        operation: tl.constexpr,
        onebyone: tl.constexpr,
        scans: tl.constexpr,
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h  # trans
        HWRoute2 = neg_h * DW + neg_w  # flip
        HWRoute3 = neg_w * DH + neg_h  # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w  # flip
        HWRoute3 = HWRoute2
    elif scans == 3:
        # none; rot90; rot180==flip; rot270;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = neg_w * DH + pos_h
        HWRoute2 = neg_h * DW + neg_w
        HWRoute3 = pos_w * DH + neg_h

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC

    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)

        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y,
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))

        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(),
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None

def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        elif scans == 3:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = torch.rot90(x, 1, dims=(2, 3)).flatten(2, 3)
            y[:, 2, :, :] = torch.rot90(x, 2, dims=(2, 3)).flatten(2, 3)
            y[:, 3, :, :] = torch.rot90(x, 3, dims=(2, 3)).flatten(2, 3)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        elif scans == 3:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = torch.rot90(x, 1, dims=(1, 2)).flatten(1, 2)
            y[:, :, 2, :] = torch.rot90(x, 2, dims=(1, 2)).flatten(1, 2)
            y[:, :, 3, :] = torch.rot90(x, 3, dims=(1, 2)).flatten(1, 2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
        elif scans == 3:
            oy = y[:, 0, :, :].contiguous().view(B, D, -1)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 1, :, :, :], -1, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, H, W)[:, 2, :, :, :], -2, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 3, :, :, :], -3, dims=(2, 3)).flatten(2, 3)
            y = oy
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)
        elif scans == 3:
            oy = y[:, :, 0, :].contiguous().view(B, -1, D)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 1, :], -1, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, H, W, K, D)[:, :, :, 2, :], -2, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 3, :], -3, dims=(1, 2)).flatten(1, 2)
            y = oy

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y
def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 3:
            y = torch.stack([
                x[:, 0, :, :, :].flatten(2, 3),
                torch.rot90(x[:, 1, :, :, :], 1, dims=(2, 3)).flatten(2, 3),
                torch.rot90(x[:, 2, :, :, :], 2, dims=(2, 3)).flatten(2, 3),
                torch.rot90(x[:, 3, :, :, :], 3, dims=(2, 3)).flatten(2, 3),
            ], dim=1)

    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)
        elif scans == 3:
            y = torch.stack([
                x[:, :, :, 0, :].flatten(1, 2),
                torch.rot90(x[:, :, :, 1, :], 1, dims=(1, 2)).flatten(1, 2),
                torch.rot90(x[:, :, :, 2, :], 2, dims=(1, 2)).flatten(1, 2),
                torch.rot90(x[:, :, :, 3, :], 3, dims=(1, 2)).flatten(1, 2),
            ], dim=1)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y
def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
        elif scans == 3:
            y = torch.stack([
                y[:, 0, :, :].contiguous().view(B, D, -1),
                torch.rot90(y.view(B, K, D, W, H)[:, 1, :, :, :], -1, dims=(2, 3)).flatten(2, 3),
                torch.rot90(y.view(B, K, D, H, W)[:, 2, :, :, :], -2, dims=(2, 3)).flatten(2, 3),
                torch.rot90(y.view(B, K, D, W, H)[:, 3, :, :, :], -3, dims=(2, 3)).flatten(2, 3),
            ], dim=1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)
        elif scans == 3:
            y = torch.stack([
                y[:, :, 0, :].contiguous().view(B, -1, D),
                torch.rot90(y.view(B, W, H, K, D)[:, :, :, 1, :], -1, dims=(1, 2)).flatten(1, 2),
                torch.rot90(y.view(B, H, W, K, D)[:, :, :, 2, :], -2, dims=(1, 2)).flatten(1, 2),
                torch.rot90(y.view(B, W, H, K, D)[:, :, :, 3, :], -3, dims=(1, 2)).flatten(1, 2),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y
class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None

class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(),
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y,
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, H, W)
            else:
                x = x.view(B, H, W, 4, C)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)

        return x, None, None, None, None
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    #todo
    # CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    CMF = CrossMergeF
    if y.is_cuda:
        with torch.cuda.device(y.device):
            return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
    else:
        return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    # CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    #todo
    CSF = CrossScanF
    if x.is_cuda:
        with torch.cuda.device(x.device):
            return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
    else:
        return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        # backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        # backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        # backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        # ctx.backend = backend
        # if backend == "oflex":
        #     out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        # elif backend == "mamba":
        #     out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        u = u.float().contiguous()
        A = A.float().contiguous()
        B = B.float().contiguous()
        C = C.float().contiguous()
        D = D.float().contiguous()
        delta = delta.float().contiguous()
        delta_bias = delta_bias.float().contiguous()
        # print(u.shape, A.shape)
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        # backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # if backend == "oflex":
        #     du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
        #         u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        #     )
        # elif backend == "mamba":
        #     du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
        #         u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
        #         False
        #     )
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_torch(
        u: torch.Tensor,  # (B, K * C, L)
        delta: torch.Tensor,  # (B, K * C, L)
        A: torch.Tensor,  # (K * C, N)
        B: torch.Tensor,  # (B, K, N, L)
        C: torch.Tensor,  # (B, K, N, L)
        D: torch.Tensor = None,  # (K * C)
        delta_bias: torch.Tensor = None,  # (K * C)
        delta_softplus=True,
        oflex=True,
        *args,
        **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2)  # (B, C, L)

    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)
def selective_scan_fn(
        u: torch.Tensor,  # (B, K * C, L)
        delta: torch.Tensor,  # (B, K * C, L)
        A: torch.Tensor,  # (K * C, N)
        B: torch.Tensor,  # (B, K, N, L)
        C: torch.Tensor,  # (B, K, N, L)
        D: torch.Tensor = None,  # (K * C)
        delta_bias: torch.Tensor = None,  # (K * C)
        delta_softplus=True,
        oflex=True,
        backend=None,
):

    # fn = selective_scan_torch if backend == "torch" or (not WITH_SELECTIVESCAN_MAMBA) else SelectiveScanCuda.apply
    fn =  SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError
class Linear(nn.Linear):
    def __init__(self, *args, channel_first=False, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups

    def forward(self, x: torch.Tensor):
        if self.channel_first:
            # B, C, H, W = x.shape
            if len(x.shape) == 4:
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif len(x.shape) == 3:
                return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)
        else:
            return F.linear(x, self.weight, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        self_state_dict = self.state_dict()
        load_state_dict_keys = list(state_dict.keys())
        if prefix + "weight" in load_state_dict_keys:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view_as(self_state_dict["weight"])
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x=x.to(torch.float32)
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x.to(orig_type)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# class VSSBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels: int = 0,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
#             # =============================
#             ssm_d_state: int = 16,
#             ssm_ratio=2.0,
#             ssm_rank_ratio=2.0,
#             ssm_dt_rank: Any = "auto",
#             ssm_act_layer=nn.SiLU,
#             ssm_conv: int = 3,
#             ssm_conv_bias=True,
#             ssm_drop_rate: float = 0,
#             ssm_init="v0",
#             forward_type="v2",
#             # =============================
#             mlp_ratio=4.0,
#             mlp_act_layer=nn.GELU,
#             mlp_drop_rate: float = 0.0,
#             # =============================
#             use_checkpoint: bool = False,
#             post_norm: bool = False,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ssm_branch = ssm_ratio > 0
#         self.mlp_branch = mlp_ratio > 0
#         self.use_checkpoint = use_checkpoint
#         self.post_norm = post_norm
#
#         # proj
#         self.proj_conv = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(hidden_dim),
#             nn.SiLU()
#         )
#
#         if self.ssm_branch:
#             self.norm = norm_layer(hidden_dim)
#             self.op = SS2D(
#                 d_model=hidden_dim,
#                 d_state=ssm_d_state,
#                 ssm_ratio=ssm_ratio,
#                 ssm_rank_ratio=ssm_rank_ratio,
#                 dt_rank=ssm_dt_rank,
#                 act_layer=ssm_act_layer,
#                 # ==========================
#                 d_conv=ssm_conv,
#                 conv_bias=ssm_conv_bias,
#                 # ==========================
#                 dropout=ssm_drop_rate,
#                 # bias=False,
#                 # ==========================
#                 # dt_min=0.001,
#                 # dt_max=0.1,
#                 # dt_init="random",
#                 # dt_scale="random",
#                 # dt_init_floor=1e-4,
#                 initialize=ssm_init,
#                 # ==========================
#                 forward_type=forward_type,
#             )
#
#         self.drop_path = DropPath(drop_path)#Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#         self.lsblock = LSBlock(hidden_dim, hidden_dim)
#         if self.mlp_branch:
#             self.norm2 = norm_layer(hidden_dim)
#             mlp_hidden_dim = int(hidden_dim * mlp_ratio)
#             self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
#                                drop=mlp_drop_rate, channels_first=False)
#
#     def forward(self, input: torch.Tensor):
#         input = self.proj_conv(input)
#         X1 = self.lsblock(input)
#         x = input + self.drop_path(self.op(self.norm(X1)))
#         if self.mlp_branch:
#             x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
#         return x
class LearnableFrequencyModulator(nn.Module):
    """
    轻量化可学习频域调制器 (Learnable Frequency Modulator)
    集成特性:
    1. 尺度不变性 (Scale Invariance): 采用双线性插值动态适配 YOLO 多尺度输入。
    2. 混合精度安全 (AMP Safe): 强制 FFT 计算在 FP32 空间进行，防止特征消失。
    """

    def __init__(self, channels, base_h=128, base_w=128):
        super().__init__()

        self.channels = channels

        # 物理限制: 实数快速傅里叶变换 (rfft2) 会根据 Nyquist 采样定理截断一半的冗余频率
        self.base_freq_w = base_w // 2 + 1

        # 频率先验掩膜 (Frequency Prior Mask)
        # 维度: (通道数, 基础高度, 截断频率宽度, 实虚部)
        # 采用实数定义以规避 PyTorch 优化器对 Complex64 参数的梯度更新 Bug
        self.complex_weight = nn.Parameter(
            torch.randn(channels, base_h, self.base_freq_w, 2, dtype=torch.float32) * 0.02
        )

        # 跨域特征融合 (Cross-domain Feature Fusion)
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        # 1. AMP 精度隔离 (Precision Isolation)
        # 红外高频噪声的振幅极小，半精度 (FP16) 的 rfft2 极易发生数值截断 (Underflow)
        # 强制将输入转换为 FP32 执行频域计算
        original_dtype = x.dtype
        x_float = x.float()
        spatial_residual = x_float

        # 获取当前动态批次的准确空域尺寸
        _, _, H, W = x_float.shape

        # 2. 空域 -> 频域变换 (Spatial to Frequency)
        # freq_x_float shape: (B, C, H_freq, W_freq)
        # 其中 H_freq = H, W_freq = W // 2 + 1
        freq_x_float = torch.fft.rfft2(x_float, norm='ortho')
        _, _, H_freq, W_freq = freq_x_float.shape

        # 3. 维度重排与插值准备 (Dimension Rearrangement)
        # F.interpolate 仅支持 (N, C, H, W) 格式的 4D 张量，不支持复数
        # 转换路径: (C, Base_H, Base_W, 2) -> (2, C, Base_H, Base_W)
        weight_permuted = self.complex_weight.permute(3, 0, 1, 2).contiguous()
        # 将实虚部合并到通道维度以欺骗插值函数: (1, 2*C, Base_H, Base_W)
        weight_reshaped = weight_permuted.view(1, self.channels * 2, self.complex_weight.size(1),
                                               self.complex_weight.size(2))

        # 4. 动态双线性插值 (Dynamic Bilinear Interpolation)
        # 使得固定的物理先验掩膜 完美拉伸/压缩 至当前特征图的频谱尺寸
        if H_freq != self.complex_weight.size(1) or W_freq != self.complex_weight.size(2):
            weight_interp = F.interpolate(
                weight_reshaped,
                size=(H_freq, W_freq),
                mode='bilinear',
                align_corners=True  # 确保低频原点(四角)对齐，不发生频移
            )
        else:
            weight_interp = weight_reshaped

        # 5. 重构复数张量 (Reconstruct Complex Tensor)
        # (1, 2*C, H_freq, W_freq) -> (2, C, H_freq, W_freq) -> (C, H_freq, W_freq, 2)
        weight_restored = weight_interp.view(2, self.channels, H_freq, W_freq).permute(1, 2, 3, 0).contiguous()
        complex_weight = torch.view_as_complex(weight_restored)

        # 6. 频域闭环调制 (Closed-loop Modulation)
        freq_modulated = freq_x_float * complex_weight

        # 7. 频域 -> 空域逆变换 (Frequency to Spatial)
        # 强制传入原始空域尺寸 s=(H, W)，防止奇数宽度的精度坍塌
        spatial_from_freq = torch.fft.irfft2(freq_modulated, s=(H, W), norm='ortho')

        # 8. 完美对齐与融合 (Alignment and Fusion)
        fused_features = torch.cat([spatial_residual, spatial_from_freq], dim=1)
        out = self.fusion_conv(fused_features)

        # 恢复网络最初的数据类型，无缝衔接下游的 FP16 卷积层
        return out.to(original_dtype)
class TaskAlignedPredictor(nn.Module):
    """
    任务对齐预测器 (Task-Aligned Predictor, TAP)
    用于在分类和回归分支解耦前，计算对齐特征与空间注意力权重，纠正 SSM 带来的特征漂移。
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # 特征降维与交互准备
        self.interact_conv = Conv(in_channels, hidden_channels, 3)

        # 预测空间对齐图 (Spatial Alignment Map)
        # 输出单通道特征图，表示每个空间像素点上分类与回归任务的对齐置信度
        self.alignment_map = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid()  # 将对齐权重归一化到 (0, 1)
        )

    def forward(self, x):
        # 提取交互特征
        interact_feat = self.interact_conv(x)
        # 计算空间对齐权重
        align_weight = self.alignment_map(interact_feat)
        # 利用注意力机制重新校准特征，抑制发生空间漂移的劣质特征点
        aligned_feat = interact_feat * align_weight
        return aligned_feat
class CCCBlock(nn.Module):
    def __init__(
            self,
            inchannel: int = 0,
            # outchannel: int = 0,
            hidden_dim: int = 96,
            drop_path: float = 0.1,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0.1,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.inchannel = inchannel
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2D1conv(#SS2D1conv,SS2D4conv,SS2Dv2,SS2D1conv_wosfi
                inchannel=inchannel,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)
        # VSSBlock(
        #     hidden_dim=96,
        #     drop_path=0.1,
        #     channel_first=False,
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v2",
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     use_checkpoint=False,
        # )
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,   act_layer=mlp_act_layer,#out_features=hidden_dim,
                           drop=mlp_drop_rate, channel_first=channel_first)
        self.first_reshape = nn.Conv2d(inchannel,hidden_dim,1,1)
        # self.saam = TaskAlignedPredictor(hidden_dim,hidden_dim)
    def _forward(self, input: torch.Tensor):
        x = self.first_reshape(input)
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                # x = self.drop_path(self.op(self.norm(x)))
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
                # x = self.drop_path(self.mlp(self.norm2(x)))
        # x = self.saam(x)
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class CCCmBlock(nn.Module):
    def __init__(
            self,
            inchannel: int = 0,
            # outchannel: int = 0,
            hidden_dim: int = 96,
            fftl: int = 0,
            drop_path: float = 0.1,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0.1,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.inchannel = inchannel
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = fSS2D1conv(#SS2D1conv,SS2D4conv,SS2Dv2
                inchannel=inchannel,
                d_model=hidden_dim,
                fftl=fftl,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)
        # VSSBlock(
        #     hidden_dim=96,
        #     drop_path=0.1,
        #     channel_first=False,
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v2",
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     use_checkpoint=False,
        # )
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,   act_layer=mlp_act_layer,#out_features=hidden_dim,
                           drop=mlp_drop_rate, channel_first=channel_first)
        self.first_reshape = nn.Conv2d(inchannel,hidden_dim,1,1)
        self.TAP = TaskAlignedPredictor(hidden_dim,hidden_dim)
    def _forward(self, input: torch.Tensor):
        x = self.first_reshape(input)
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                # x = self.drop_path(self.op(self.norm(x)))
                x = x * self.drop_path(self.op(self.norm(x)))
        # x = self.TAP(x)
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
                # x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
class VSSBlock(nn.Module):
    def __init__(
            self,
            inchannel: int = 0,
            # outchannel: int = 0,
            hidden_dim: int = 96,
            drop_path: float = 0.1,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.inchannel = inchannel
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2Dv2(
                inchannel=inchannel,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)
        # VSSBlock(
        #     hidden_dim=96,
        #     drop_path=0.1,
        #     channel_first=False,
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v2",
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     use_checkpoint=False,
        # )
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,   act_layer=mlp_act_layer,#out_features=hidden_dim,
                           drop=mlp_drop_rate, channel_first=channel_first)
        self.first_reshape = nn.Conv2d(inchannel,hidden_dim,1,1)
        self.saam = TaskAlignedPredictor(hidden_dim, hidden_dim)
    def _forward(self, input: torch.Tensor):
        x = self.first_reshape(input)
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                # x = self.drop_path(self.op(self.norm(x)))
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
                # x = self.drop_path(self.mlp(self.norm2(x)))
        x = self.saam(x)
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
#todo 1scan convcancat
class LearnableFFTLayerFull(nn.Module):
    def __init__(self, d, l):
        super(LearnableFFTLayerFull, self).__init__()
        self.l = l

        # 权重维度：(通道数 d, 序列全长 l)
        # 使用复数参数，直接调整幅度和相位
        self.complex_weight = nn.Parameter(
            torch.randn(d, l, dtype=torch.complex64) * 0.02
        )

    def forward(self, x):
        # x 维度: (b, d, l)

        # 1. 全长度快速傅里叶变换
        # 输出维度: (b, d, l), 类型: complex64
        x_freq = torch.fft.fft(x, dim=-1)

        # 2. 频域点乘
        # (b, d, l) * (d, l) -> (b, d, l)
        x_freq = x_freq * self.complex_weight

        # 3. 逆傅里叶变换
        # 输出维度: (b, d, l), 类型: complex64
        x_res = torch.fft.ifft(x_freq, dim=-1)

        # 4. 取实部回到原尺寸实数空间
        # 注意：如果输入本身是复数，则不应调用 .real
        return torch.real(x_res)
class SS2D1conv(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            inchannel,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.inchannel = inchannel
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo

        # nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.my_conv = nn.Conv2d(self.d_inner,self.d_inner,3,groups=self.d_inner,padding=1)
        self.my_conv = nn.Sequential(*(Conv(self.d_inner, self.d_inner, 3, g=self.d_inner, p=1) for _ in range(1)))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        # self.fft = LearnableFrequencyModulator(d_proj,80,80 )
        # self.fft = LearnableFFTLayerFull(self.d_inner,51200)
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            #todo

            # x_con = self.fft(x)
            x_con = self.my_conv(x)

            # x_con = conv_norm(x_con)
            x_con = x_con.flatten(2, 3)
            # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            x_cons = x.new_empty((B, self.d_inner, H * W, 2))
            x_cons[:, :, :, 0] = xs[:, 0, :, :]
            x_cons[:, :, :, 1] = x_con
            x_conmulty = x_cons.flatten(2, 3)
            # try:
            #     x_conmulty = self.fft(x_conmulty)
            # except:
            #     pass
            x_consmulty = x.new_empty((B, 1, self.d_inner, 2 * H * W))
            x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_consmulty, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L*2)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L*2)
            dts = dts.contiguous().view(B, -1, L*2)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L*2)
            Cs = Cs.contiguous().view(B, K, N, L*2)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                x_conmulty, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            ys = (ys[:, :, ::2]+ys[:, :, 1::2]).contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        # y = self.fft(y)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        # y = self.fft(y)
        out = self.dropout(self.out_proj(y))


        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
class SS2D1conv_wosfi(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            inchannel,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.inchannel = inchannel
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo

        # nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.my_conv = nn.Conv2d(self.d_inner,self.d_inner,3,groups=self.d_inner,padding=1)
        self.my_conv = nn.Sequential(*(Conv(self.d_inner, self.d_inner, 3, g=self.d_inner, p=1) for _ in range(1)))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        # self.fft = LearnableFrequencyModulator(d_proj,80,80 )
        # self.fft = LearnableFFTLayerFull(self.d_inner,51200)
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            #todo

            # x_con = self.fft(x)
            # x_con = self.my_conv(x)

            # x_con = conv_norm(x_con)
            # x_con = x_con.flatten(2, 3)
            # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            x_cons = x.new_empty((B, self.d_inner, H * W, 1))
            x_cons[:, :, :, 0] = xs[:, 0, :, :]
            # x_cons[:, :, :, 1] = x_con
            x_conmulty = x_cons.flatten(2, 3)
            # try:
            #     x_conmulty = self.fft(x_conmulty)
            # except:
            #     pass
            x_consmulty = x.new_empty((B, 1, self.d_inner,  H * W))
            x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_consmulty, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                x_conmulty, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            # ys = (ys[:, :, ::2]+ys[:, :, 1::2]).contiguous().view(B, K, -1, H, W)
            ys = ys.contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        # y = self.fft(y)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        # y = self.fft(y)
        out = self.dropout(self.out_proj(y))


        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
class TaskAlignedPredictor(nn.Module):
    """
    任务对齐预测器 (Task-Aligned Predictor, TAP)
    用于在分类和回归分支解耦前，计算对齐特征与空间注意力权重，纠正 SSM 带来的特征漂移。
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # 特征降维与交互准备
        self.interact_conv = Conv(in_channels, hidden_channels, 3)

        # 预测空间对齐图 (Spatial Alignment Map)
        # 输出单通道特征图，表示每个空间像素点上分类与回归任务的对齐置信度
        self.alignment_map = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid()  # 将对齐权重归一化到 (0, 1)
        )

    def forward(self, x):
        # 提取交互特征
        interact_feat = self.interact_conv(x)
        # 计算空间对齐权重
        align_weight = self.alignment_map(interact_feat)
        # 利用注意力机制重新校准特征，抑制发生空间漂移的劣质特征点
        aligned_feat = interact_feat * align_weight
        return aligned_feat
class fSS2D1conv(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            inchannel,
            d_model=96,
            fftl=0,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.inchannel = inchannel
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo

        # nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.my_conv = nn.Conv2d(self.d_inner,self.d_inner,3,groups=self.d_inner,padding=1)
        self.my_conv = nn.Sequential(*(nn.Conv2d(self.d_inner, self.d_inner, 3, groups=self.d_inner, padding=1) for _ in range(4)))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        # self.fft = LearnableFrequencyModulator(d_proj,80,80 )
        # self.fft = LearnableFFTLayerFull(d_proj,fftl)
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            #todo

            # x_con = self.fft(x)
            x_con = self.my_conv(x)

            # x_con = conv_norm(x_con)
            x_con = x_con.flatten(2, 3)
            # x_con = self.fft(x_con)
            # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            x_cons = x.new_empty((B, self.d_inner, H * W, 2))
            x_cons[:, :, :, 0] = xs[:, 0, :, :]
            x_cons[:, :, :, 1] = x_con
            x_conmulty = x_cons.flatten(2, 3)
            x_consmulty = x.new_empty((B, 1, self.d_inner, 2 * H * W))
            x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_consmulty, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L*2)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L*2)
            dts = dts.contiguous().view(B, -1, L*2)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L*2)
            Cs = Cs.contiguous().view(B, K, N, L*2)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                x_conmulty, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            ys = (ys[:, :, ::2]+ys[:, :, 1::2]).contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


#4scan convcancat
class SS2D4conv(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            inchannel,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.inchannel = inchannel
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo
        self.my_conv = nn.Conv2d(self.d_inner,self.d_inner,3,groups=self.d_inner,padding=1)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="cross2d",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            #todo
            x_con = self.my_conv(x)
            # x_con = conv_norm(x_con)
            x_con = cross_scan_fn(x_con, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            # x_con = x_con.flatten(2, 3)
            # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            x_cons = x.new_empty((B, self.k_group, self.d_inner, H * W, 2))
            x_cons[:, :, :, :, 0] = xs[:, :, :, :]
            x_cons[:, :, :, :, 1] = x_con[:, :, :, :]
            x_conmulty = x_cons.flatten(3, 4)
            # x_consmulty = x.new_empty((B, 1, self.d_inner, 2 * H * W))
            # x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_conmulty, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L*2)
            dts = self.dt_projs(dts)

            x_conmulty = x_conmulty.view(B, -1, L*2)
            dts = dts.contiguous().view(B, -1, L*2)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L*2)
            Cs = Cs.contiguous().view(B, K, N, L*2)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                x_conmulty, dts, Bs, Cs = to_fp32(x_conmulty, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                x_conmulty, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            ys = ys[:, :, ::2].contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


# MutiFeatureMix
class MutiFeatureMixSSM(nn.Module):
    def __init__(
            self,
            inchannel: [int] ,
            # outchannel: int = 0,
            hidden_dim: int = 128,#应等于此前dwconv个数
            drop_path: float = 0.1,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,#todo
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 1,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        # self.inchannel = inchannel
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2DguideMix(#SS2D1conv,SS2D4conv,SS2Dv2
                # inchannel=inchannel,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)
        # VSSBlock(
        #     hidden_dim=96,
        #     drop_path=0.1,
        #     channel_first=False,
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v2",
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     use_checkpoint=False,
        # )
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,   act_layer=mlp_act_layer,#out_features=hidden_dim,
                           drop=mlp_drop_rate, channel_first=channel_first)
        # self.first_reshape = nn.Conv2d(4,hidden_dim,1,1)
        self.reshape = nn.Conv2d(inchannel[1], inchannel[0], kernel_size=1, bias=False)
        # self.haar=HaarWaveletDownsampling(inchannel[1], inchannel[0], scale_factor=int(inchannel[0]/inchannel[1]))#scale_factor
    def _forward(self, input: [torch.Tensor]):
        # xt = torch.chunk(input,chunks= self.inchannel,dim=1)
        x0 = input[0]

        x1 = input[1]
        # x1 = self.haar(x1)
        x1 = self.reshape(x1)

        x = [x0, x1]
        x = [self.norm(i) for i in x]
                # x = self.drop_path(self.op(self.norm(x)))
        x = x[0] + self.drop_path(self.op(x))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
                # x = self.drop_path(self.mlp(self.norm2(x)))
        return x
    def forward(self, input: [torch.Tensor]):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
#channel wise
class HaarWaveletDownsamplingSize(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=8):
        super().__init__()
        self.scale_factor = scale_factor
        self.levels = int((math.log2(scale_factor))/2)
        poolsize = 2 ** self.levels
        # poolsize = 2 * 2 ** self.levels
        self.pool = nn.AvgPool2d(kernel_size=poolsize, stride=poolsize, padding=0)
        # final_dim = in_channels * (4 ** self.levels)
        # self.channel_reducer = nn.Conv2d(final_dim, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('haar_weights', self._create_haar_kernel(in_channels))
    def _create_haar_kernel(self, channels):
        weights = torch.zeros(channels * 4, channels, 2, 2)
        for c in range(channels):
            # LL (Low-Low)
            weights[c * 4 + 0, c, :, :] = 0.5 * torch.tensor([[1, 1], [1, 1]])
            # LH (Low-High) - Vertical Edges
            weights[c * 4 + 1, c, :, :] = 0.5 * torch.tensor([[-1, -1], [1, 1]])
            # HL (High-Low) - Horizontal Edges
            weights[c * 4 + 2, c, :, :] = 0.5 * torch.tensor([[-1, 1], [-1, 1]])
            # HH (High-High) - Diagonal Edges
            weights[c * 4 + 3, c, :, :] = 0.5 * torch.tensor([[1, -1], [-1, 1]])
        return weights
    def dwt_step(self, x, current_channels):
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, h // 2, 2, w // 2, 2)
        x_00 = x_reshaped[:, :, :, 0, :, 0]

        # 提取右上 (Row=0, Col=1)
        x_01 = x_reshaped[:, :, :, 0, :, 1]

        # 提取左下 (Row=1, Col=0)
        x_10 = x_reshaped[:, :, :, 1, :, 0]

        # 提取右下 (Row=1, Col=1)
        x_11 = x_reshaped[:, :, :, 1, :, 1]

        # Haar 变换公式
        ll = (x_00 + x_01 + x_10 + x_11) / 2
        lh = (x_00 + x_01 - x_10 - x_11) / 2
        hl = (x_00 - x_01 + x_10 - x_11) / 2
        hh = (x_00 - x_01 - x_10 + x_11) / 2

        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        out = x
        curr_c = x.shape[1]

        # 递归执行 DWT (3次)
        for _ in range(self.levels):
            out = self.dwt_step(out, curr_c)
            curr_c *= 4

            # 此时 out shape: [1, 3072, 16, 16]
        # 降维对齐
        # out = self.channel_reducer(out)
        out = self.pool(out)

        out = self.bn(out)
        return out

class HaarWaveletDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=8):
        super().__init__()
        self.scale_factor = scale_factor
        self.levels = int(math.log2(scale_factor))
        final_dim = in_channels * (4 ** self.levels)
        self.channel_reducer = nn.Conv2d(final_dim, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('haar_weights', self._create_haar_kernel(in_channels))
    def _create_haar_kernel(self, channels):
        weights = torch.zeros(channels * 4, channels, 2, 2)
        for c in range(channels):
            # LL (Low-Low)
            weights[c * 4 + 0, c, :, :] = 0.5 * torch.tensor([[1, 1], [1, 1]])
            # LH (Low-High) - Vertical Edges
            weights[c * 4 + 1, c, :, :] = 0.5 * torch.tensor([[-1, -1], [1, 1]])
            # HL (High-Low) - Horizontal Edges
            weights[c * 4 + 2, c, :, :] = 0.5 * torch.tensor([[-1, 1], [-1, 1]])
            # HH (High-High) - Diagonal Edges
            weights[c * 4 + 3, c, :, :] = 0.5 * torch.tensor([[1, -1], [-1, 1]])
        return weights
    def dwt_step(self, x, current_channels):
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, h // 2, 2, w // 2, 2)
        x_00 = x_reshaped[:, :, :, 0, :, 0]

        # 提取右上 (Row=0, Col=1)
        x_01 = x_reshaped[:, :, :, 0, :, 1]

        # 提取左下 (Row=1, Col=0)
        x_10 = x_reshaped[:, :, :, 1, :, 0]

        # 提取右下 (Row=1, Col=1)
        x_11 = x_reshaped[:, :, :, 1, :, 1]

        # Haar 变换公式
        ll = (x_00 + x_01 + x_10 + x_11) / 2
        lh = (x_00 + x_01 - x_10 - x_11) / 2
        hl = (x_00 - x_01 + x_10 - x_11) / 2
        hh = (x_00 - x_01 - x_10 + x_11) / 2

        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        out = x
        curr_c = x.shape[1]

        # 递归执行 DWT (3次)
        for _ in range(self.levels):
            out = self.dwt_step(out, curr_c)
            curr_c *= 4

            # 此时 out shape: [1, 3072, 16, 16]
        # 降维对齐
        out = self.channel_reducer(out)
        out = self.bn(out)
        return out

class SS2DguideMix(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            # inchannel,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=1,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # self.inchannel = inchannel
        self.inchannel = 2
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        # self.fft = LearnableFrequencyModulator(self.d_inner, 320, 320)
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)


        B, D, H, W = x[0].shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xst = [cross_scan_fn(i, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch) for i in x]
            #todo
            xs = x[0].new_empty(B, self.d_inner, L, self.inchannel)

            for j in range(self.inchannel):
                xs[:, :, :, j] = xst[j][:,0]
            xs = xs.flatten(2, 3)
            txs = x[0].new_empty((B, 1, self.d_inner, self.inchannel * L))
            txs[:, 0] = xs
            # x_con = self.my_conv(x)
            # # x_con = conv_norm(x_con)
            # x_con = x_con.flatten(2, 3)
            # # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            # x_cons = x.new_empty((B, self.d_inner, H * W, 2))
            # x_cons[:, :, :, 0] = xs[:, 0, :, :]
            # x_cons[:, :, :, 1] = x_con
            # x_conmulty = x_cons.flatten(2, 3)
            # x_consmulty = x.new_empty((B, 1, self.d_inner, 2 * H * W))
            # x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", txs, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L*self.inchannel)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L*self.inchannel)
            dts = dts.contiguous().view(B, -1, L*self.inchannel)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L*self.inchannel)
            Cs = Cs.contiguous().view(B, K, N, L*self.inchannel)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            yst: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            ys = yst.view(B, -1, L, self.inchannel).sum(dim=3)
            # for ti in range(self.inchannel):
            #     ys+=yst[:, :, ti::self.inchannel]
            ys=ys.contiguous().view(B, K, -1, H, W)
            # ys = (ys[:, :, ::self.inchannel] + ys[:, :, 1::self.inchannel]).contiguous().view(B, K, -1, H, W)
            # ys = ys[:, :, ::2].contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x[0].dtype)

    def forward(self, x: torch.Tensor, **kwargs):

        x = [self.in_proj(x[i]) for i in range(self.inchannel)]
        # x = self.in_proj(x)
        # if not self.disable_z:
        #     x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
        #     if not self.disable_z_act:
        #         z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # if self.with_dconv:
        #     x = self.conv2d(x)  # (b, d, h, w)
        # x = self.act(x)
        x = [self.act(x[i]) for i in range(self.inchannel)]
        y = self.forward_core(x)
        y = self.out_act(y)
        # if not self.disable_z:
        #     y = y * z
        # y =self.fft(y)
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value