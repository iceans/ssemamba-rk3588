# -*- coding: utf-8 -*-
"""SpecTransSSM 冒烟测试: 数学正确性 + 工程可用性"""
import torch, math
from spec_trans_ssm import (SpecTransSSM, SpectralTransport,
                            linear_scan_parallel, linear_scan_sequential)

torch.manual_seed(0)
OK = []

# 1. 并行扫描 == 序贯递归 (复数, 随机 a,b)
a = torch.exp(torch.complex(-torch.rand(2, 16, 4, 8, 5), torch.randn(2, 16, 4, 8, 5)))
b = torch.complex(torch.randn(2, 16, 4, 8, 5), torch.randn(2, 16, 4, 8, 5))
hp = linear_scan_parallel(a.clone(), b.clone())
hs = linear_scan_sequential(a, b)
err = (hp - hs).abs().max().item()
assert err < 1e-4, f"scan mismatch {err}"
OK.append(f"1. 并行扫描==序贯递归  max|Δ|={err:.2e}")

# 2. 相位=平移: 纯对流一步应把脉冲搬运 v 个像素 (整数 v 严格验证)
st = SpectralTransport(d_state=1, selective=False, use_diff=False,
                       use_decay=False, pad=0)
with torch.no_grad():
    # 强制常数系数: vx=3, vy=-2  (tanh 反解)
    c = st.coeff
    c.const.zero_()
    c.bias.zero_()
    c.const[0] = math.atanh(3.0 / 4.0)   # vx
    c.const[1] = math.atanh(-2.0 / 4.0)  # vy
H = W = 32
seq = torch.zeros(2, 1, H, W)          # B=1, T=2: t0 注入脉冲, t1 零注入
seq[0, 0, 10, 10] = 1.0                # h_1 = 脉冲; h_2 = a·h_1 → 被搬运
h, _, _ = st(seq, seq, B=1, T=2)
h2 = h.view(1, 2, 1, H, W)[0, 1, 0]
iy, ix = divmod(h2.argmax().item(), W)
assert (iy, ix) == (8, 13), f"advection moved to {(iy, ix)}, expect (8,13)"
peak = h2[8, 13].item()
assert abs(peak - 1.0) < 1e-4, f"peak={peak} (应无插值损耗)"
OK.append(f"2. 相位=平移: (10,10)+v(3,-2) → ({iy},{ix}), 峰值={peak:.6f} (零插值损耗)")

# 2b. 半群性质 (可扫描性的物理体现): shift(0.5)∘shift(0.5) == shift(1) 精确成立
with torch.no_grad():
    c.const[0] = math.atanh(0.5 / 4.0); c.const[1] = 0.0
seq3 = torch.zeros(3, 1, H, W); seq3[0, 0, 10, 10] = 1.0
h, _, _ = st(seq3, seq3, 1, 3)
h3 = h.view(1, 3, 1, H, W)[0, 2, 0]           # 经过两次 v=0.5 转移
iy, ix = divmod(h3.argmax().item(), W)
assert (iy, ix) == (10, 11) and abs(h3[10, 11].item() - 1.0) < 1e-4
assert abs(h3.sum().item() - 1.0) < 1e-4       # DC 保持 ⇒ 能量守恒
OK.append(f"2b. 半群复合: 0.5px×2 == 1px 精确 (峰值 {h3[10,11]:.6f}, 总能量 {h3.sum():.6f})")

# 3. 无条件稳定: 随机系数下逐频率 |a|<=1, 100 步滚动无爆炸
st2 = SpectralTransport(d_state=8, selective=True, pad=0)
uu = torch.randn(4, 8, 16, 16)
a3, _ = st2.transition(uu, B=1, T=4, Hp=16, Wp=16)
assert a3.abs().max().item() <= 1.0 + 1e-6
m = SpecTransSSM(c1=8, d_state=8, seq_len=1, mode="spectral", pad=0)
m.streaming = True; m.reset_state(); m.eval()
mx = 0.0
with torch.no_grad():
    for _ in range(100):
        y = m(torch.randn(1, 8, 24, 24))
        mx = max(mx, m._state.abs().max().item())
assert mx < 1e3 and not math.isnan(mx)
OK.append(f"3. |a(k)|≤1 断言通过; 100 帧流式滚动 max|ĥ|={mx:.2f} (无 CFL, 无爆炸)")

# 4. 四种 mode 前向/反向 + 形状
for mode in ("spectral", "local", "hybrid", "diag"):
    net = SpecTransSSM(c1=16, d_state=16, seq_len=4, mode=mode, pad=4)
    x = torch.randn(8, 16, 20, 20, requires_grad=True)  # B=2, T=4
    y = net(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
OK.append("4. spectral/local/hybrid/diag 前向反向通过, 形状保持, 梯度有限")

# 5. 零初始化 = 恒等 (热插拔两阶段训练的前提)
net = SpecTransSSM(c1=16, d_state=16, seq_len=4, mode="hybrid", pad=4)
x = torch.randn(8, 16, 20, 20)
d = (net(x) - x).abs().max().item()
assert d < 1e-5, f"init not identity: {d}"
OK.append(f"5. 零初始化输出=输入 (max|Δ|={d:.1e}), 热插拔成立")

# 6. 流式 == 整段 (spectral, 序贯扫描口径), 训练 T / 推理任意长解耦
net = SpecTransSSM(c1=8, d_state=8, seq_len=6, mode="spectral",
                   pad=0, scan="parallel").eval()
with torch.no_grad():
    for p in net.out_proj.parameters():
        p.normal_(0, 0.1)  # 打破恒等以便比较
    xs = torch.randn(6, 8, 16, 16)  # B=1,T=6 整段
    y_batch = net(xs)
    net.streaming = True; net.reset_state()
    y_stream = torch.cat([net(xs[i:i + 1]) for i in range(6)])
e = (y_batch - y_stream).abs().max().item()
assert e < 1e-4, f"stream mismatch {e}"
OK.append(f"6. 并行扫描整段 == 逐帧流式  max|Δ|={e:.2e} (常数显存流式成立)")

# 7. N % T != 0 的回退 (单帧验证/热插拔) 不崩
net = SpecTransSSM(c1=8, d_state=8, seq_len=8, mode="spectral", pad=0)
_ = net(torch.randn(3, 8, 16, 16))
OK.append("7. batch 不整除 seq_len 时回退 T=1, 单帧验证兼容")

print("\n".join("[PASS] " + s for s in OK))

# 8. 完整新息门 (M4 严格版, local/hybrid 序贯) 前向反向; spectral+innov 正确拒绝
for mode in ("local", "hybrid"):
    net = SpecTransSSM(c1=16, d_state=16, seq_len=4, mode=mode, gate="innov", pad=4)
    x = torch.randn(8, 16, 20, 20, requires_grad=True)
    y = net(x); y.sum().backward()
    assert torch.isfinite(x.grad).all()
try:
    SpecTransSSM(c1=8, d_state=8, seq_len=4, mode="spectral", gate="innov")
    raise RuntimeError("should have refused")
except AssertionError:
    pass
OK.append("8. 新息门逐步闭环 (预测->新息->增益->更新) 通过; spectral+innov 按设计拒绝")
print("[PASS] " + OK[-1])
