"""End-to-end detection FPS on RK3588 (int8), single core.
Compares full-DFL decode vs prefiltered decode, on a real video stream.
"""
import time
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_int8.rknn"
VIDEO = "/home/Tronlong/RK3588_uav/1.mp4"
IMG_SIZE = 640
CONF, IOU = 0.25, 0.45   # 与板子历史评估一致
N = 300


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def decode_full(outputs):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2 * i], np.float32)
        lg = np.asarray(outputs[2 * i + 1], np.float32)
        n, c, h, w = box.shape
        y = box.reshape(n, 4, c // 4, h, w)
        e = np.exp(y - y.max(2, keepdims=True))
        y = e / e.sum(2, keepdims=True)
        dist = (y * np.arange(c // 4).reshape(1, 1, c // 4, 1, 1)).sum(2)
        col, row = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.concatenate((col.reshape(1, 1, h, w), row.reshape(1, 1, h, w)), 1)
        s = np.array([IMG_SIZE // w, IMG_SIZE // h]).reshape(1, 2, 1, 1)
        xyxy = np.concatenate(((grid + 0.5 - dist[:, 0:2]) * s, (grid + 0.5 + dist[:, 2:4]) * s), 1)
        boxes.append(xyxy.reshape(4, -1).T)
        scores.append(sig(lg).reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)


def decode_fast(outputs, conf=CONF):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2 * i], np.float32)
        lg = np.asarray(outputs[2 * i + 1], np.float32)
        h, w = lg.shape[2], lg.shape[3]
        cls = sig(lg).reshape(-1)
        idx = np.nonzero(cls >= conf)[0]
        if idx.size == 0:
            continue
        bf = box[0].reshape(4, 16, h, w).transpose(2, 3, 0, 1).reshape(h * w, 4, 16)[idx]
        e = np.exp(bf - bf.max(2, keepdims=True))
        p = e / e.sum(2, keepdims=True)
        dist = (p * np.arange(16)).sum(2)
        gy, gx = idx // w, idx % w
        st = IMG_SIZE // w
        boxes.append(np.stack([(gx + 0.5 - dist[:, 0]) * st, (gy + 0.5 - dist[:, 1]) * st,
                               (gx + 0.5 + dist[:, 2]) * st, (gy + 0.5 + dist[:, 3]) * st], 1))
        scores.append(cls[idx])
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.concatenate(boxes), np.concatenate(scores)


def nms(boxes, scores, thr=IOU):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1 + np.where(ovr <= thr)[0]]
    return np.array(keep, dtype=int)


def run(decode_fn, label):
    cap = cv2.VideoCapture(VIDEO)
    r = RKNNLite(); r.load_rknn(MODEL); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    for _ in range(5):
        ok, fr = cap.read()
        xx = cv2.cvtColor(cv2.resize(fr, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
        r.inference(inputs=[xx], data_format=["nhwc"])
    n = 0; tr = tp = ti = td = 0.0
    t0 = time.perf_counter()
    while n < N:
        a = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        b = time.perf_counter(); tr += b - a
        xx = cv2.cvtColor(cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR),
                          cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
        c = time.perf_counter(); tp += c - b
        o = r.inference(inputs=[xx], data_format=["nhwc"])
        d = time.perf_counter(); ti += d - c
        boxes, scores = decode_fn(o)
        if len(boxes):
            k = nms(boxes, scores)
            for bb in boxes[k]:
                cv2.rectangle(frame, (int(bb[0] * frame.shape[1] / 640), int(bb[1] * frame.shape[0] / 640)),
                              (int(bb[2] * frame.shape[1] / 640), int(bb[3] * frame.shape[0] / 640)), (0, 0, 255), 2)
        e = time.perf_counter(); td += e - d
        n += 1
    dt = time.perf_counter() - t0
    print(f"[{label}] frames={n} total={dt:.2f}s -> {n/dt:.1f} FPS (single core, video 1080p)")
    print(f"    read {tr/n*1000:.2f} | pre {tp/n*1000:.2f} | infer {ti/n*1000:.2f} | decode+nms+draw {td/n*1000:.2f} ms")
    cap.release(); r.release()


if __name__ == "__main__":
    print("video:", VIDEO)
    run(decode_full, "full-DFL decode")
    run(decode_fast, "prefilter decode")
