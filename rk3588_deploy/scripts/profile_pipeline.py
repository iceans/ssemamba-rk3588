"""End-to-end detection pipeline profiling on RK3588 (int8).
Measures each stage: imread / preprocess / NPU infer / DFL decode / sigmoid+NMS / draw.
"""
import time
import glob
import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_int8.rknn"
IMG = "/home/Tronlong/RK3588_uavtest/detect_results/0302_detect.jpg"
IMG_SIZE = 640
IOU = 0.45
CONF = 0.5
N = 200


def dfl(pos):
    n, c, h, w = pos.shape
    p_num, mc = 4, c // 4
    y = pos.reshape(n, p_num, mc, h, w)
    e = np.exp(y - y.max(2, keepdims=True))
    y = e / e.sum(2, keepdims=True)
    acc = np.arange(mc).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def decode(outputs):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2 * i], dtype=np.float32)
        lg = np.asarray(outputs[2 * i + 1], dtype=np.float32)
        cls = 1.0 / (1.0 + np.exp(-lg))
        n, c, h, w = box.shape
        pos = dfl(box)
        col, row = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.concatenate((col.reshape(1, 1, h, w), row.reshape(1, 1, h, w)), 1)
        s = np.array([IMG_SIZE // w, IMG_SIZE // h]).reshape(1, 2, 1, 1)
        xyxy = np.concatenate(((grid + 0.5 - pos[:, 0:2]) * s, (grid + 0.5 + pos[:, 2:4]) * s), 1)
        boxes.append(xyxy.reshape(4, -1).T)
        scores.append(cls.reshape(-1))
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


def timeit(fn, n=N):
    for _ in range(5):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000.0


def main():
    r = RKNNLite()
    assert r.load_rknn(MODEL) == 0
    assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0

    frame = cv2.imread(IMG)
    print("source image:", frame.shape, "(BGR)")

    pre = lambda: cv2.cvtColor(cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
    x = pre()
    infer = lambda: r.inference(inputs=[x], data_format=["nhwc"])
    outs = infer()

    def dec():
        b, s = decode(outs)
        return b, s
    boxes, scores = dec()

    def post():
        m = scores >= CONF
        b, s = boxes[m], scores[m]
        return b[nms(b, s)] if len(b) else b

    # component timings
    t_read = timeit(lambda: cv2.imread(IMG))
    t_pre = timeit(pre)
    t_infer = timeit(infer)
    t_dec = timeit(dec)
    t_post = timeit(post)

    print("\n--- 单阶段耗时(ms, 单核, {}次平均) ---".format(N))
    print(f"imread (磁盘读图)        : {t_read:7.2f}")
    print(f"preprocess (resize+cvtColor+uint8): {t_pre:6.2f}")
    print(f"NPU infer (rknn.inference): {t_infer:7.2f}")
    print(f"DFL decode (softmax+grid) : {t_dec:7.2f}")
    print(f"sigmoid+NMS               : {t_post:7.2f}")
    print(f"--- 纯NPU合计            : {t_infer:7.2f} ms -> {1000/t_infer:.1f} FPS")
    print(f"--- 端到端(不含draw/显示)  : {t_read+t_pre+t_infer+t_dec+t_post:7.2f} ms -> "
          f"{1000/(t_read+t_pre+t_infer+t_dec+t_post):.1f} FPS")

    # full pipeline incl draw
    def full():
        f = cv2.imread(IMG)
        xx = cv2.cvtColor(cv2.resize(f, (640, 640), interpolation=cv2.INTER_LINEAR),
                          cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
        o = r.inference(inputs=[xx], data_format=["nhwc"])
        b, s = decode(o)
        m = s >= CONF
        b, s = b[m], s[m]
        if len(b):
            k = nms(b, s)
            for bb in b[k]:
                cv2.rectangle(f, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
        return f

    t_full = timeit(full)
    print(f"--- 完整单帧(读+预处理+推理+解码+NMS+画框, 无显示): {t_full:7.2f} ms -> {1000/t_full:.1f} FPS")

    r.release()


if __name__ == "__main__":
    main()
