"""Full DAUB test-set evaluation on RK3588 (fp16 RKNN), 3-core parallel.
Saves predictions (original pixel coords) as JSON: {image_name: [[x1,y1,x2,y2,conf],...]}
"""
import glob
import json
import os
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_fp16.rknn"
IMG_DIR = "/home/Tronlong/rknn_deploy/images"
OUT = "/home/Tronlong/rknn_deploy/preds.json"
IMG_SIZE = 640
REG_MAX = 16
STRIDE = [8, 16, 32]
CONF = 0.001
IOU = 0.7


def dfl(pos):
    n, c, h, w = pos.shape
    p_num, mc = 4, c // 4
    y = pos.reshape(n, p_num, mc, h, w)
    e = np.exp(y - y.max(2, keepdims=True))
    y = e / e.sum(2, keepdims=True)
    acc = np.arange(mc).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)  # [1,4,h,w]


def decode(outputs):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2 * i], dtype=np.float32)      # [1,64,h,w]
        lg = np.asarray(outputs[2 * i + 1], dtype=np.float32)   # [1,1,h,w]
        cls = 1.0 / (1.0 + np.exp(-lg))                         # float32 sigmoid
        n, c, h, w = box.shape
        pos = dfl(box)
        col, row = np.meshgrid(np.arange(w), np.arange(h))
        col = col.reshape(1, 1, h, w)
        row = row.reshape(1, 1, h, w)
        grid = np.concatenate((col, row), 1)
        s = np.array([IMG_SIZE // w, IMG_SIZE // h]).reshape(1, 2, 1, 1)
        xy1 = (grid + 0.5 - pos[:, 0:2]) * s
        xy2 = (grid + 0.5 + pos[:, 2:4]) * s
        xyxy = np.concatenate((xy1, xy2), 1).reshape(4, -1).T    # [hw,4]
        boxes.append(xyxy)
        scores.append(cls.reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)


def nms(boxes, scores, iou_thr=IOU):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1 + np.where(ovr <= iou_thr)[0]]
    return np.array(keep, dtype=int)


def make_rknn(core):
    r = RKNNLite()
    assert r.load_rknn(MODEL) == 0
    assert r.init_runtime(core_mask=core) == 0
    return r


def process(path, rknn):
    img = cv2.imread(path)
    oh, ow = img.shape[:2]
    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float16)[None]  # [1,640,640,3] in [0,255]
    outs = rknn.inference(inputs=[x], data_format=["nhwc"])
    boxes, scores = decode(outs)
    m = scores >= CONF
    boxes, scores = boxes[m], scores[m]
    if boxes.shape[0]:
        keep = nms(boxes, scores)
        boxes, scores = boxes[keep], scores[keep]
    sx, sy = ow / IMG_SIZE, oh / IMG_SIZE
    boxes = boxes * np.array([sx, sy, sx, sy])
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)]
            for b, s in zip(boxes, scores)]


def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.bmp")))
    print("images:", len(files))
    rknns = [make_rknn(c) for c in (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)]

    def worker(idx_path):
        idx, path = idx_path
        return os.path.basename(path), process(path, rknns[idx % 3])

    t0 = time.time()
    preds = {}
    done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for name, p in ex.map(worker, enumerate(files)):
            preds[name] = p
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(files)}  {time.time()-t0:.1f}s", flush=True)

    with open(OUT, "w") as f:
        json.dump(preds, f)
    n_box = sum(len(v) for v in preds.values())
    print(f"done: {len(preds)} images, {n_box} boxes, {time.time()-t0:.1f}s")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
