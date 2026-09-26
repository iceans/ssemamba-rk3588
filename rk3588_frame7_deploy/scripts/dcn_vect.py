import torch, torch.nn.functional as F

def deform_conv2d_forward(self, input, offset, mask=None):
    """Vectorized DCNv2, groups=1. Matches torchvision.ops.DeformConv2d."""
    B, C, H, W = input.shape
    Co = self.weight.shape[0]
    kH, kW = self.weight.shape[2], self.weight.shape[3]
    pad_h, pad_w = self.padding if isinstance(self.padding, (tuple, list)) else (self.padding, self.padding)
    str_h, str_w = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride, self.stride)
    dil_h, dil_w = self.dilation if isinstance(self.dilation, (tuple, list)) else (self.dilation, self.dilation)
    Hout = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // str_h + 1
    Wout = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // str_w + 1
    K2 = kH * kW
    dev = input.device
    hh = torch.arange(Hout, device=dev, dtype=input.dtype)
    ww = torch.arange(Wout, device=dev, dtype=input.dtype)
    ki = torch.arange(K2, device=dev, dtype=input.dtype) // kW
    kj = torch.arange(K2, device=dev, dtype=input.dtype) % kW
    # base coords, shape (K2,Hout,Wout)
    by = ki.view(K2, 1, 1) * dil_h + hh.view(1, Hout, 1) * str_h - pad_h
    bx = kj.view(K2, 1, 1) * dil_w + ww.view(1, 1, Wout) * str_w - pad_w
    off = offset.view(B, K2, 2, Hout, Wout).permute(0, 1, 3, 4, 2)   # (B,K2,Hout,Wout,2) -> [dy,dx]
    gx = (bx.view(1, K2, 1, Wout) + off[..., 1]) / (W - 1) * 2 - 1
    gy = (by.view(1, K2, Hout, 1) + off[..., 0]) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(B, K2 * Hout, Wout, 2)
    s = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    s = s.view(B, C, K2, Hout, Wout)
    if mask is not None:
        s = s * mask.view(B, 1, K2, Hout, Wout)
    s = s.reshape(B, C * K2, Hout * Wout)
    w = self.weight.reshape(Co, C * K2)
    out = torch.matmul(w, s).reshape(B, Co, Hout, Wout)
    if self.bias is not None:
        out = out + self.bias.view(1, Co, 1, 1)
    return out

if __name__ == '__main__':
    import torchvision.ops as ops, types
    torch.manual_seed(0)
    for k, pad, st in [(3, 1, 1), (1, 0, 1), (3, 1, 2)]:
        conv = ops.DeformConv2d(5, 6, kernel_size=k, padding=pad, stride=st, bias=True)
        x = torch.randn(2, 5, 16, 16)
        Hout = (16 + 2 * pad - (k - 1) - 1) // st + 1
        off = torch.randn(2, 2 * k * k, Hout, Hout) * 0.5
        m = torch.rand(2, k * k, Hout, Hout)
        ref = conv(x, off, mask=m)
        conv.forward = types.MethodType(deform_conv2d_forward, conv)
        got = conv(x, off, mask=m)
        print('k', k, 'pad', pad, 'stride', st, 'shapes', tuple(ref.shape), tuple(got.shape),
              'maxdiff', float((ref - got).abs().max()), 'refmax', float(ref.abs().max()))
