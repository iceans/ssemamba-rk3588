"""Preprocess one DAUB test image + compute PyTorch split-head outputs."""
import types
import numpy as np
import cv2
import torch
from ultralytics import YOLO

PT = "/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/yolov8n_813/weights/best.pt"
IMG = "/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format/test/images/IR_00006_00000.bmp"
SIZE = 640
OUT = "/home/dell/lxs/rk3588_deploy/"


def split_forward(self, x):
    outs = []
    for i in range(self.nl):
        outs.append(self.cv2[i](x[i]))
        outs.append(self.cv3[i](x[i]))
    return outs


img = cv2.imread(IMG)
img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

nchw = img.transpose(2, 0, 1)[None]          # [1,3,640,640] float32
nhwc = img[None]                              # [1,640,640,3] float32
np.save(OUT + "input_nchw.npy", nchw)
np.save(OUT + "input_nhwc.npy", nhwc)

model = YOLO(PT)
net = model.model
net.eval()
detect = net.model[-1]
detect.forward = types.MethodType(split_forward, detect)

with torch.no_grad():
    outs = net(torch.from_numpy(nchw))

names = ["box0", "cls0", "box1", "cls1", "box2", "cls2"]
d = {}
for n, o in zip(names, outs):
    a = o.cpu().numpy().astype(np.float32)
    d[n] = a
    print(n, a.shape, a.min(), a.max())
np.savez(OUT + "pt_out.npz", **d)
print("saved pt_out.npz")
