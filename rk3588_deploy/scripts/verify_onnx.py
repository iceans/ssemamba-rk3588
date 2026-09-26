"""ONNX outputs via onnxruntime."""
import numpy as np
import onnxruntime as ort

OUT = "/home/dell/lxs/rk3588_deploy/"
x = np.load(OUT + "input_nchw.npy")

sess = ort.InferenceSession(OUT + "best_split.onnx", providers=["CPUExecutionProvider"])
outs = sess.run(None, {"images": x})

names = ["box0", "cls0", "box1", "cls1", "box2", "cls2"]
d = {}
for n, o in zip(names, outs):
    a = o.astype(np.float32)
    d[n] = a
    print(n, a.shape, a.min(), a.max())
np.savez(OUT + "onnx_out.npz", **d)
print("saved onnx_out.npz")
