"""Decode + NMS compare for pt/onnx/rknn on the single test image."""
import numpy as np

OUT = "/home/dell/lxs/rk3588_deploy/"
STRIDE = [8, 16, 32]
REG_MAX = 16
IMG = 640


def dfl(pos):
    n, c, h, w = pos.shape
    p_num = 4
    mc = c // p_num
    y = pos.reshape(n, p_num, mc, h, w)
    e = np.exp(y - y.max(2, keepdims=True))
    y = e / e.sum(2, keepdims=True)
    acc = np.arange(mc).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def decode(d):
    boxes, scores = [], []
    for i in range(3):
        box = d[f"box{i}"]              # [1,64,h,w]
        lg = d[f"cls{i}"].reshape(-1)   # raw logits
        cls = 1.0 / (1.0 + np.exp(-lg)) # float32 sigmoid
        n, c, h, w = box.shape
        pos = dfl(box)                  # [1,4,h,w]
        col, row = np.meshgrid(np.arange(w), np.arange(h))
        col = col.reshape(1, 1, h, w); row = row.reshape(1, 1, h, w)
        grid = np.concatenate((col, row), 1)
        s = np.array([IMG // w, IMG // h]).reshape(1, 2, 1, 1)
        xy1 = (grid + 0.5 - pos[:, 0:2]) * s
        xy2 = (grid + 0.5 + pos[:, 2:4]) * s
        xyxy = np.concatenate((xy1, xy2), 1)      # [1,4,h,w] x1,y1,x2,y2
        boxes.append(xyxy.reshape(4, -1).T)       # [hw,4]
        scores.append(cls)                        # [hw]
    boxes = np.concatenate(boxes)                 # [8400,4]
    scores = np.concatenate(scores)               # [8400]
    return boxes, scores


def nms(boxes, scores, iou_thr=0.7):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1 + np.where(iou <= iou_thr)[0]]
    return np.array(keep)


def report(name, d, conf=0.001):
    boxes, scores = decode(d)
    m = scores >= conf
    b, s = boxes[m], scores[m]
    keep = nms(b, s)
    b, s = b[keep], s[keep]
    top = s.argsort()[::-1][:5]
    print(f"\n{name}: {len(b)} boxes after NMS (conf>={conf})")
    for i in top:
        print(f"  conf={s[i]:.4f} box=({b[i][0]:.1f},{b[i][1]:.1f},{b[i][2]:.1f},{b[i][3]:.1f})")
    return b, s


pt = np.load(OUT + "pt_out.npz")
on = np.load(OUT + "onnx_out.npz")
rk = np.load(OUT + "rknn_out.npz")

b_pt, s_pt = report("PyTorch", pt)
b_on, s_on = report("ONNX", on)
b_rk, s_rk = report("RKNN-fp16", rk)

if len(b_pt) and len(b_rk):
    print("\nbest box pt:", b_pt[0].round(1), "conf", round(s_pt[0], 4))
    print("best box rk:", b_rk[0].round(1), "conf", round(s_rk[0], 4))
    print("box diff (px):", np.abs(b_pt[0] - b_rk[0]).round(2), "conf diff:", round(abs(s_pt[0] - s_rk[0]), 4))
