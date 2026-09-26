"""RKNN fp16 outputs on board, correct input mode: [0,255] fp16 NHWC."""
import numpy as np
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_fp16.rknn"
OUT = "/home/Tronlong/rknn_deploy/"

img = np.load("/home/Tronlong/rknn_deploy/input_nhwc.npy")      # [0,1] float32
x = (img * 255.0).astype(np.float16)                            # [0,255] fp16

r = RKNNLite()
assert r.load_rknn(MODEL) == 0
assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
for _ in range(3):
    r.inference(inputs=[x], data_format=["nhwc"])
outs = r.inference(inputs=[x], data_format=["nhwc"])

names = ["box0", "cls0", "box1", "cls1", "box2", "cls2"]
d = {}
for n, o in zip(names, outs):
    a = np.asarray(o, dtype=np.float32)
    d[n] = a
    print(n, a.shape, a.min(), a.max())
np.savez(OUT + "rknn_out.npz", **d)
print("saved rknn_out.npz")
r.release()
