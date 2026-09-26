"""Compare pt / onnx / rknn outputs."""
import numpy as np

OUT = "/home/dell/lxs/rk3588_deploy/"
pt = np.load(OUT + "pt_out.npz")
on = np.load(OUT + "onnx_out.npz")
rk = np.load(OUT + "rknn_out.npz")

names = ["box0", "cls0", "box1", "cls1", "box2", "cls2"]

def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print(f"{'tensor':<6} {'pt vs onnx maxabs':>18} {'pt vs rknn maxabs':>18} {'pt vs rknn cos':>14}")
ok = True
for n in names:
    d_po = np.abs(pt[n] - on[n]).max()
    d_pr = np.abs(pt[n] - rk[n]).max()
    c_pr = cos(pt[n], rk[n])
    flag = ""
    if n.startswith("box"):
        thresh = 0.5  # box logits: fp16 abs err budget
    else:
        thresh = 0.02  # sigmoid outputs in [0,1]
    if d_pr > thresh or c_pr < 0.99:
        flag = "  <-- CHECK"
        ok = False
    print(f"{n:<6} {d_po:>18.6f} {d_pr:>18.6f} {c_pr:>14.6f}{flag}")

print("\nVERIFY:", "PASS" if ok else "FAIL")
