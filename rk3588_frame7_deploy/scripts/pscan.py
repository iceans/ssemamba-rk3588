"""Exact associative (Hillis-Steele) parallel selective scan with chunking.

Replacement for the CUDA-only SelectiveScanCuda in
ultralytics/nn/modules/tgsmamba.py. Exports to ONNX using only
Pad/Slice/Mul/Add/ReduceSum/Exp/Reshape.

Why chunking: the plain parallel scan pads the length to 1.5*L while shifting;
for L=5*320*320=512000 that creates tensors with a 774144-wide dimension, which
exceeds the RK3588 NPU max dimension (~65535) and makes inference fail with
"failed to submit!".  Chunking reshapes the sequence into (n_chunks, chunk) so
every tensor dimension stays below `chunk*1.5` and `n_chunks`.
"""
import torch
import torch.nn.functional as F


def _shift(x, steps, fill, dim):
    if steps == 0:
        return x
    D = x.dim()
    dim = dim % D
    L = x.shape[dim]
    pad = [0] * (2 * D)
    pad[2 * (D - 1 - dim)] = steps
    xp = F.pad(x, pad, mode="constant", value=fill)
    idx = [slice(None)] * D
    idx[dim] = slice(0, L)
    return xp[tuple(idx)]


def _scan(a, b, dim):
    """Inclusive associative scan of (a,b) along `dim`."""
    step = 1
    L = a.shape[dim]
    while step < L:
        a_s = _shift(a, step, 1.0, dim)
        b_s = _shift(b, step, 0.0, dim)
        b = b + a * b_s
        a = a * a_s
        step *= 2
    return a, b


def _pick_chunk(L, target):
    """Largest divisor of L that is <= target (so no padding is needed)."""
    for d in range(min(target, L), 0, -1):
        if L % d == 0:
            return d
    return 0


def selective_scan_parallel(u, delta, A, B, C, D=None, delta_bias=None,
                            delta_softplus=True, oflex=True, chunk=24576,
                            *args, **kwargs):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.log1p(torch.exp(torch.clamp(delta, max=10.0)))  # 10 -> fp16-safe

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)

    a = torch.exp(delta.unsqueeze(-1) * A.view(1, KCdim, 1, N))        # (b,d,l,n)
    b = delta.unsqueeze(-1) * u.unsqueeze(-1) * B.permute(0, 1, 3, 2)   # (b,d,l,n)

    Ch = _pick_chunk(L, chunk) if chunk else 0
    if Ch and Ch < L:
        num = L // Ch
        a = a.reshape(Batch, KCdim, num, Ch, N)
        b = b.reshape(Batch, KCdim, num, Ch, N)
        # 1) intra-chunk inclusive scan along chunk axis (dim=3)
        ai, bi = _scan(a, b, 3)
        A_chunk = ai[..., -1, :]      # (b,d,num,n)
        B_chunk = bi[..., -1, :]
        # 2) inclusive scan over chunks (dim=2); need exclusive incoming state
        Ac, Bc = _scan(A_chunk, B_chunk, 2)
        state = _shift(Bc, 1, 0.0, 2)  # state entering each chunk (exclusive)
        # 3) combine
        x = bi + ai * state.unsqueeze(3)
        x = x.reshape(Batch, KCdim, L, N)
    else:
        _, x = _scan(a, b, 2)

    C = C.permute(0, 1, 3, 2)      # (b,d,l,n)
    y = (x * C).sum(dim=-1)        # (b,d,l)
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/dell/lxs/tsgmamba/ultralytics-main")
    from ultralytics.nn.modules.tgsmamba import selective_scan_torch
    torch.manual_seed(0)

    def ref(u, delta, A, B, C, D, db):
        return selective_scan_torch(u, delta, A, B, C, D, db, True)

    # small: chunked vs loop reference
    Bn, K, N, L, Cdim = 2, 2, 3, 257, 4
    KC = K * Cdim
    u = torch.randn(Bn, KC, L); delta = torch.rand(Bn, KC, L) + 0.05
    A = -torch.rand(KC, N) - 0.1
    Bt = torch.randn(Bn, K, N, L); Ct = torch.randn(Bn, K, N, L)
    db = torch.randn(KC); D = torch.randn(KC)
    for ch in (0, 64, 1000):
        r = selective_scan_parallel(u, delta, A, Bt, Ct, D, db, True, chunk=ch)
        rr = ref(u, delta, A, Bt, Ct, D, db)
        print(f"small chunk={ch:5d} maxdiff={(r-rr).abs().max().item():.3e}")

    # large: chunked vs unchunked parallel scan (reference loop too slow)
    Bn, K, N, L, Cdim = 1, 1, 16, 100000, 8
    KC = K * Cdim
    u = torch.randn(Bn, KC, L); delta = torch.rand(Bn, KC, L) + 0.01
    A = -torch.rand(KC, N) - 0.1
    Bt = torch.randn(Bn, K, N, L); Ct = torch.randn(Bn, K, N, L)
    r_full = selective_scan_parallel(u, delta, A, Bt, Ct, None, None, True, chunk=0)
    for ch in (32768, 16384):
        r_ch = selective_scan_parallel(u, delta, A, Bt, Ct, None, None, True, chunk=ch)
        print(f"large L={L} chunk={ch:6d} maxdiff={(r_ch-r_full).abs().max().item():.3e} "
              f"absmax={r_full.abs().max().item():.3f}")
