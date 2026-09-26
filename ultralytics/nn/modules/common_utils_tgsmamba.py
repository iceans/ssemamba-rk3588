
import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
# class CrossScan(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor):
#         B, C, H, W = x.shape
#         ctx.shape = (B, C, H, W)
#         xs = x.new_empty((B, 4, C, H * W))
#         xs[:, 0] = x.flatten(2, 3)
#         xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
#         xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
#         return xs
#
#     @staticmethod
#     def backward(ctx, ys: torch.Tensor):
#         # out: (b, k, d, l)
#         B, C, H, W = ctx.shape
#         L = H * W
#         ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
#         y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
#         return y.view(B, -1, H, W)
#todo no trans my test

# class CrossScan(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor):
#         B, C, H, W = x.shape
#         ctx.shape = (B, C, H, W)
#         xs = x.new_empty((B, 2, C, H * W))#2channle
#         xs[:, 0] = x.flatten(2, 3)
#         # xs[:, 1] = x.flatten(2, 3)
#         xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
#
#
#         return xs
#
#     @staticmethod
#     def backward(ctx, ys: torch.Tensor):
#         # out: (b, k, d, l)
#         B, C, H, W = ctx.shape
#         L = H * W
#         y = ys[:, 0] + ys[:, 1].flip(dims=[-1])
#         # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).contiguous().view(B, -1, L)
# #         ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
# #         y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
#         return y.view(B, -1, H, W)
#todo no trans no flip

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))#2channle
        xs[:, 0] = x.flatten(2, 3)
        # xs[:, 1] = x.flatten(2, 3)
        # xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])


        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0] #+ ys[:, 1].flip(dims=[-1])
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).contiguous().view(B, -1, L)
#         ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
#         y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)

# class CrossMerge(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, ys: torch.Tensor):
#         B, K, D, H, W = ys.shape
#         ctx.shape = (H, W)
#         ys = ys.view(B, K, D, -1)
#         ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
#         y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
#         return y
#
#     @staticmethod
#     def backward(ctx, x: torch.Tensor):
#         # B, D, L = x.shape
#         # out: (b, k, d, l)
#         H, W = ctx.shape
#         B, C, L = x.shape
#         xs = x.new_empty((B, 4, C, L))
#         xs[:, 0] = x
#         xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
#         xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
#         xs = xs.view(B, 4, C, H, W)
#         return xs, None, None
#todo no trans,my test

# class CrossMerge(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, ys: torch.Tensor):
#         B, K, D, H, W = ys.shape
#         ctx.shape = (H, W)
#         ys = ys.view(B, K, D, -1)
#         y = ys[:, 0] + ys[:, 1].flip(dims=[-1])#.view(B, 1, D, -1)
#         # y = ys[:, 0] + ys[:, 1].contiguous()
#         return y
#
#     @staticmethod
#     def backward(ctx, x: torch.Tensor):
#         # B, D, L = x.shape
#         # out: (b, k, d, l)
#         H, W = ctx.shape
#         B, C, L = x.shape
#         xs = x.new_empty((B, 2, C, L))
#         xs[:, 0] = x
#         # xs[:, 1] = x.view(B, C, H, W).flatten(2, 3)
#         xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
#         xs = xs.view(B, 2, C, H, W)
#         return xs, None, None

#todo no trans,no flip, my test

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] #+ ys[:, 1].flip(dims=[-1])#.view(B, 1, D, -1)
        # y = ys[:, 0] + ys[:, 1].contiguous()
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        ## xs[:, 1] = x.view(B, C, H, W).flatten(2, 3)
        #xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 1, C, H, W)
        return xs, None, None
# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)



#todo conv plus

# def cross_selective_scan(
#         x: torch.Tensor = None,
#         x_proj_weight: torch.Tensor = None,
#         x_proj_bias: torch.Tensor = None,
#         dt_projs_weight: torch.Tensor = None,
#         dt_projs_bias: torch.Tensor = None,
#         A_logs: torch.Tensor = None,
#         Ds: torch.Tensor = None,
#         out_norm: torch.nn.Module = None,
#         out_norm_shape="v0",
#         nrows=-1,  # for SelectiveScanNRow
#         backnrows=-1,  # for SelectiveScanNRow
#         delta_softplus=True,
#         to_dtype=True,
#         force_fp32=False,  # False if ssoflex
#         ssoflex=True,
#         SelectiveScan=None,
#         scan_mode_type='default',
#         my_conv:torch.Tensor = None,
#         conv_num = None,
#         # conv_norm:torch.Tensor = None,
#         dt_norm_map_weight:torch.Tensor = None#引用单独注释
# ):
#     # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
#     B, D, H, W = x.shape
#     D, N = A_logs.shape
#     K, D, R = dt_projs_weight.shape
#     L = H * W
#     def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
#         return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
#     # TOD
#     xs = CrossScan.apply(x)
#     #
#     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)#批，方向，矩阵相乘。proj矩阵预测Δ，B，C
#     if x_proj_bias is not None:
#         x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
#     dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
#     dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

#     #TOD conv addition
#     x_con = my_conv(x)
#     # x_con = conv_norm(x_con)
#     x_con = x_con.flatten(2, 3)
#
#     # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
#     x_cons = x.new_empty((B, K, conv_num, H * W))
#     x_cons[:, 0] = x_con
#     # x_cons[:, 1] = x_con.flip(dims=[-1])
#     # xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
#     # xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
#     x_cons = nn.functional.normalize(x_cons, p = 2, dim = -1)
#     dts_con, Bs_con, Cs_con = torch.split(x_cons, [D, N, N], dim=2)# B C dlt
#     # dts_con, Cs_con = torch.split(x_cons, [D, N], dim=2)  # too BC
#     # Bs_con = x_cons#tdo BC
#     # Cs_con = x_cons  # to BC
#     # TOO plus miu
#     miu=0.5
#     dts = nn.functional.normalize(dts, p=2, dim=-1)
#     # dts_con = nn.functional.normalize(dts_con, p=2, dim=-1)
#     Bs = nn.functional.normalize(Bs, p=2, dim=-1)
#     Cs = nn.functional.normalize(Cs, p=2, dim=-1)
#     # dts_cons = nn.functional.normalize(dts_cons, p=2, dim=-1)
#     dts = miu * dts + (1 - miu) * dts_con
#     Bs = miu * Bs + (1 - miu) * Bs_con
#     Cs = miu * Cs + (1 - miu) * Cs_con
#     dts = nn.functional.relu(dts)
#     Bs = nn.functional.relu(Bs)
#     Cs = nn.functional.relu(Cs)
#     # dts = nn.functional.dropout(dts,0.5)
#     # end
#     xs = xs.view(B, -1, L)
#     dts = dts.contiguous().view(B, -1, L)
#     # HiPPO matrix
#     As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
#     Bs = Bs.contiguous()
#     Cs = Cs.contiguous()
#     Ds = Ds.to(torch.float)  # (K * c)
#     delta_bias = dt_projs_bias.view(-1).to(torch.float)
#     if force_fp32:
#         xs = xs.to(torch.float)
#         dts = dts.to(torch.float)
#         Bs = Bs.to(torch.float)
#     ys: torch.Tensor = selective_scan(
#         xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
#     ).view(B, K, -1, H, W)
#     y: torch.Tensor = CrossMerge.apply(ys)
#     if out_norm_shape in ["v1"]:  # (B, C, H, W)
#         y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
#     else:  # (B, L, C)
#         y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
#         y = out_norm(y).view(B, H, W, -1)
#     return (y.to(x.dtype) if to_dtype else y)


#todo multyscale sptial token catch, half lines stay STtran k4

# def cross_selective_scan(
#         x: torch.Tensor = None,
#         x_proj_weight: torch.Tensor = None,
#         x_proj_bias: torch.Tensor = None,
#         dt_projs_weight: torch.Tensor = None,
#         dt_projs_bias: torch.Tensor = None,
#         A_logs: torch.Tensor = None,
#         Ds: torch.Tensor = None,
#         out_norm: torch.nn.Module = None,
#         out_norm_shape="v0",
#         nrows=-1,  # for SelectiveScanNRow
#         backnrows=-1,  # for SelectiveScanNRow
#         delta_softplus=True,
#         to_dtype=True,
#         force_fp32=False,  # False if ssoflex
#         ssoflex=True,
#         SelectiveScan=None,
#         scan_mode_type='default',
#         my_conv:torch.Tensor = None,
#         conv_num = None,
#         # conv_norm:torch.Tensor = None,
#         dt_norm_map_weight:torch.Tensor = None#引用单独注释
# ):
#     # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
#     B, D, H, W = x.shape
#     D, N = A_logs.shape
#     K, D, R = dt_projs_weight.shape
#     L = H * W
#     def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
#         return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
#     # TOD
#     xs = CrossScan.apply(x)
#     #
#     # TOD conv addition
#     x_con = my_conv(x)
#     # x_con = conv_norm(x_con)
#     # x_con = x_con.flatten(2, 3)
#     x_con = CrossScan.apply(x_con)
#     # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
#     x_cons = x.new_empty((B, conv_num, H * W,2))
#     x_consmulty = x.new_empty((B, 4, conv_num, 2 * H * W))
#     x_cons[:,:,:, 0] = xs[:,0,:,:]
#     x_cons[:,:,:, 1] = x_con[:,0,:,:]
#     x_conmulty=x_cons.flatten(2, 3)
#     x_consmulty[:,0] = x_conmulty
#
#     x_cons[:, :, :, 0] = xs[:, 1, :, :]
#     x_cons[:, :, :, 1] = x_con[:,1,:,:]
#     x_conmulty = x_cons.flatten(2, 3)
#     x_consmulty[:, 1] = x_conmulty
#
#     x_cons[:, :, :, 0] = xs[:, 2, :, :]
#     x_cons[:, :, :, 1] = x_con[:,2,:,:]
#     x_conmulty = x_cons.flatten(2, 3)
#     x_consmulty[:, 2] = x_conmulty
#
#     x_cons[:, :, :, 0] = xs[:, 3, :, :]
#     x_cons[:, :, :, 1] = x_con[:,2,:,:]
#     x_conmulty = x_cons.flatten(2, 3)
#     x_consmulty[:, 3] = x_conmulty
#
#     x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_consmulty, x_proj_weight)#批，方向，矩阵相乘。proj矩阵预测Δ，B，C
#     if x_proj_bias is not None:
#         x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
#     dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
#     dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
#     x_conmulty = x_conmulty.view(B, -1, L*2)
#     dts = dts.contiguous().view(B, -1, L*2)
#     # HiPPO matrix
#     As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
#     Bs = Bs.contiguous()
#     Cs = Cs.contiguous()
#     Ds = Ds.to(torch.float)  # (K * c)
#     delta_bias = dt_projs_bias.view(-1).to(torch.float)
#     if force_fp32:
#         x_conmulty = x_conmulty.to(torch.float)
#         dts = dts.to(torch.float)
#         Bs = Bs.to(torch.float)
#     ys: torch.Tensor = selective_scan(
#         x_conmulty, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
#     ).view(B, K, -1, H, W)
#     y: torch.Tensor = CrossMerge.apply(ys)
#     if out_norm_shape in ["v1"]:  # (B, C, H, W)
#         y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
#     else:  # (B, L, C)
#         y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
#         # y = y[:,:,:conv_num]
#         y = y[:, :, ::2]
#         y = out_norm(y).view(B, H, W, -1)
#     # return y.to(torch.float32)
#     return (y.to(x.dtype) if to_dtype else y)

#todo multyscale sptial token catch, half lines stay STtran mAP50 0.632 baseline0.6

def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default',
        my_conv:torch.Tensor = None,
        conv_num = None,
        # conv_norm:torch.Tensor = None,
        dt_norm_map_weight:torch.Tensor = None#引用单独注释
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    # TOD
    xs = CrossScan.apply(x)
    #
    # TOD conv addition
    x_con = my_conv(x)
    # x_con = conv_norm(x_con)
    x_con = x_con.flatten(2, 3)
    # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
    x_cons = x.new_empty((B, conv_num, H * W,2))
    x_cons[:,:,:, 0] = xs[:,0,:,:]
    x_cons[:,:,:, 1] = x_con
    x_conmulty=x_cons.flatten(2, 3)
    x_consmulty = x.new_empty((B, 1, conv_num, 2*H * W))
    x_consmulty[:,0] = x_conmulty

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_consmulty, x_proj_weight)#批，方向，矩阵相乘。proj矩阵预测Δ，B，C
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    x_conmulty = x_conmulty.view(B, -1, L*2)
    dts = dts.contiguous().view(B, -1, L*2)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    if force_fp32:
        x_conmulty = x_conmulty.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
    ys: torch.Tensor = selective_scan(
        x_conmulty, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    y: torch.Tensor = CrossMerge.apply(ys)
    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        # y = y[:,:,:conv_num]
        y = y[:, :, ::2]
        y = out_norm(y).view(B, H, W, -1)
    # return y.to(torch.float32)
    return (y.to(x.dtype) if to_dtype else y)

#todo Bs Cs dlt on mult

# def cross_selective_scan(
#         x: torch.Tensor = None,
#         x_proj_weight: torch.Tensor = None,
#         x_proj_bias: torch.Tensor = None,
#         dt_projs_weight: torch.Tensor = None,
#         dt_projs_bias: torch.Tensor = None,
#         A_logs: torch.Tensor = None,
#         Ds: torch.Tensor = None,
#         out_norm: torch.nn.Module = None,
#         out_norm_shape="v0",
#         nrows=-1,  # for SelectiveScanNRow
#         backnrows=-1,  # for SelectiveScanNRow
#         delta_softplus=True,
#         to_dtype=True,
#         force_fp32=False,  # False if ssoflex
#         ssoflex=True,
#         SelectiveScan=None,
#         scan_mode_type='default',
#         my_conv:torch.Tensor = None,
#         conv_norm:torch.Tensor = None,
#         dt_norm_map_weight:torch.Tensor = None,#引用单独注释
#         conv_num = None
# ):
#     # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
#
#     B, D, H, W = x.shape
#     D, N = A_logs.shape
#     K, D, R = dt_projs_weight.shape
#     L = H * W
#
#     def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
#         return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
#
#     # TOD
#
#     xs = CrossScan.apply(x)
#     #
#     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)#批，方向，矩阵相乘。proj矩阵预测Δ，B，C
#     if x_proj_bias is not None:
#         x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
#     dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
#     dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
#     # #TOO conv addition
#     x_con = my_conv(x)
#     # x_con = conv_norm(x_con)
#     x_con = x_con.flatten(2, 3)
#     x_cons = x.new_empty((B, K, D+N+N, H * W))# B(C) dlt
#     # x_cons = x.new_empty((B, K,  conv_num, H * W))# B C dlt
#     # x_cons = x.new_empty((B, K, D, H * W))#  dlt
#     # x_cons = x.new_empty((B, K, N+N, H * W))# B C
#
#     x_cons[:, 0] = x_con
#
#     # # x_cons[:, 1] = x_con.flip(dims=[-1])
#
#     # x_cons[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
#     # xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
#     x_cons = nn.functional.normalize(x_cons, p = 2, dim = -1)
#     dts_con, Bs_con, Cs_con = torch.split(x_cons, [D, N, N], dim=2)#tdo BC
#     # # dts_con, Cs_con = torch.split(x_cons, [D, N], dim=2)  # tdo BC
#     # # Bs_con = x_cons#too BC
#     # # Cs_con = x_cons  # too BC
#     #TOD multy
#
#     # test
#     # dts = nn.functional.normalize(dts, p=2, dim=-1)
#     # dts = dts + 1e-6
#     dts = torch.einsum("b k d l, b a d l -> b k d l", dts, dts_con)#tdo BC
#     Bs = torch.einsum("b k d l, b a d l -> b k d l", Bs, Bs_con)#too BC
#     Cs = torch.einsum("b k d l, b a d l -> b k d l", Cs, Cs_con)#too BC
#     #end
#
#     xs = xs.view(B, -1, L)
#     dts = dts.contiguous().view(B, -1, L)
#     # HiPPO matrix
#     As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
#     Bs = Bs.contiguous()
#     Cs = Cs.contiguous()
#     Ds = Ds.to(torch.float)  # (K * c)
#     delta_bias = dt_projs_bias.view(-1).to(torch.float)
#     if force_fp32:
#         xs = xs.to(torch.float)
#         dts = dts.to(torch.float)
#         Bs = Bs.to(torch.float)
#     ys: torch.Tensor = selective_scan(
#         xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
#     ).view(B, K, -1, H, W)
#
#     y: torch.Tensor = CrossMerge.apply(ys)
#
#     if out_norm_shape in ["v1"]:  # (B, C, H, W)
#         y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
#     else:  # (B, L, C)
#         y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
#         y = out_norm(y).view(B, H, W, -1)
# #
# #     return (y.to(x.dtype) if to_dtype else y)
#
# #todo basic scan mamba
#
# def cross_selective_scan(
#         x: torch.Tensor = None,
#         x_proj_weight: torch.Tensor = None,
#         x_proj_bias: torch.Tensor = None,
#         dt_projs_weight: torch.Tensor = None,
#         dt_projs_bias: torch.Tensor = None,
#         A_logs: torch.Tensor = None,
#         Ds: torch.Tensor = None,
#         out_norm: torch.nn.Module = None,
#         out_norm_shape="v0",
#         nrows=-1,  # for SelectiveScanNRow
#         backnrows=-1,  # for SelectiveScanNRow
#         delta_softplus=True,
#         to_dtype=True,
#         force_fp32=False,  # False if ssoflex
#         ssoflex=True,
#         SelectiveScan=None,
#         scan_mode_type='default',
#         my_conv:torch.Tensor = None,
#         conv_norm:torch.Tensor = None,
#         dt_norm_map_weight:torch.Tensor = None#引用单独注释
# ):
#     # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
#     B, D, H, W = x.shape
#     D, N = A_logs.shape
#     K, D, R = dt_projs_weight.shape
#     L = H * W
#     def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
#         return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
#     xs = CrossScan.apply(x)
#     #
#     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)#批，方向，矩阵相乘。proj矩阵预测Δ，B，C
#     if x_proj_bias is not None:
#         x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
#     dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
#     dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
#     xs = xs.view(B, -1, L)
#     dts = dts.contiguous().view(B, -1, L)
#     # HiPPO matrix
#     As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
#     Bs = Bs.contiguous()
#     Cs = Cs.contiguous()
#     Ds = Ds.to(torch.float)  # (K * c)
#     delta_bias = dt_projs_bias.view(-1).to(torch.float)
#     if force_fp32:
#         xs = xs.to(torch.float)
#         dts = dts.to(torch.float)
#         Bs = Bs.to(torch.float)
#     ys: torch.Tensor = selective_scan(
#         xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
#     ).view(B, K, -1, H, W)
#     y: torch.Tensor = CrossMerge.apply(ys)
#     if out_norm_shape in ["v1"]:  # (B, C, H, W)
#         y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
#     else:  # (B, L, C)
#         y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
#         y = out_norm(y).view(B, H, W, -1)
#     return (y.to(x.dtype) if to_dtype else y)

#todo origin mamba

# def cross_selective_scan(
#         x: torch.Tensor = None,
#         x_proj_weight: torch.Tensor = None,
#         x_proj_bias: torch.Tensor = None,
#         dt_projs_weight: torch.Tensor = None,
#         dt_projs_bias: torch.Tensor = None,
#         A_logs: torch.Tensor = None,
#         Ds: torch.Tensor = None,
#         out_norm: torch.nn.Module = None,
#         out_norm_shape="v0",
#         nrows=-1,  # for SelectiveScanNRow
#         backnrows=-1,  # for SelectiveScanNRow
#         delta_softplus=True,
#         to_dtype=True,
#         force_fp32=False,  # False if ssoflex
#         ssoflex=True,
#         SelectiveScan=None,
#         scan_mode_type='default'
# ):
#     # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
#
#     B, D, H, W = x.shape
#     D, N = A_logs.shape
#     K, D, R = dt_projs_weight.shape
#     L = H * W
#
#     def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
#         return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
#
#     xs = CrossScan.apply(x)
#
#     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
#     if x_proj_bias is not None:
#         x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
#     dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
#     dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
#     xs = xs.view(B, -1, L)
#     dts = dts.contiguous().view(B, -1, L)
#     # HiPPO matrix
#     As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
#     Bs = Bs.contiguous()
#     Cs = Cs.contiguous()
#     Ds = Ds.to(torch.float)  # (K * c)
#     delta_bias = dt_projs_bias.view(-1).to(torch.float)
#
#     if force_fp32:
#         xs = xs.to(torch.float)
#         dts = dts.to(torch.float)
#         Bs = Bs.to(torch.float)
#         Cs = Cs.to(torch.float)
#
#     ys: torch.Tensor = selective_scan(
#         xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
#     ).view(B, K, -1, H, W)
#
#     y: torch.Tensor = CrossMerge.apply(ys)
#
#     if out_norm_shape in ["v1"]:  # (B, C, H, W)
#         y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
#     else:  # (B, L, C)
#         y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
#         y = out_norm(y).view(B, H, W, -1)
#
#     return (y.to(x.dtype) if to_dtype else y)