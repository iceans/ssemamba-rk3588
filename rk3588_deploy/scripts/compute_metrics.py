"""Compute YOLO-standard metrics (P/R/mAP50/mAP50-95) for RKNN predictions."""
import glob
import json
import os
import numpy as np
from ultralytics.utils.metrics import DetMetrics

PRED = "/home/dell/lxs/rk3588_deploy/preds.json"
LABEL_DIR = "/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format/test/labels"
IMG = 256
IOUV = np.linspace(0.5, 0.95, 10)


def box_iou_np(gt, det):
    """IoU between gt [M,4] and det [N,4] -> [M,N]."""
    gt = gt[:, None, :]
    det = det[None, :, :]
    x1 = np.maximum(gt[..., 0], det[..., 0])
    y1 = np.maximum(gt[..., 1], det[..., 1])
    x2 = np.minimum(gt[..., 2], det[..., 2])
    y2 = np.minimum(gt[..., 3], det[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    a_det = (det[..., 2] - det[..., 0]) * (det[..., 3] - det[..., 1])
    return inter / (a_gt + a_det - inter + 1e-12)


def match_predictions(pred_cls, true_cls, iou, iouv):
    """Replicates ultralytics DetectionValidator.match_predictions."""
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    correct_class = true_cls[:, None] == pred_cls  # [n_gt, n_pred]
    iou = iou * correct_class
    for i, thr in enumerate(iouv):
        matches = np.array(np.nonzero(iou >= thr)).T  # [k,2] (gt, pred)
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


preds = json.load(open(PRED))
files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))

tp_l, conf_l, pcls_l, tcls_l = [], [], [], []

for f in files:
    name = os.path.basename(f).replace(".txt", ".bmp")

    gts = []
    with open(f) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 5:
                cls, xc, yc, w, h = map(float, p)
                gts.append([(xc - w / 2) * IMG, (yc - h / 2) * IMG,
                            (xc + w / 2) * IMG, (yc + h / 2) * IMG, int(cls)])
    gts = np.array(gts, dtype=np.float32).reshape(-1, 5) if gts else np.zeros((0, 5))

    p = preds.get(name, [])
    if p:
        p = np.array(p, dtype=np.float32)          # [n,5] x1,y1,x2,y2,conf
        pbox, pconf, pcls = p[:, :4], p[:, 4], np.zeros(len(p), dtype=int)
    else:
        pbox = np.zeros((0, 4)); pconf = np.zeros(0); pcls = np.zeros(0, dtype=int)

    if len(pbox) and len(gts):
        iou = box_iou_np(gts[:, :4], pbox)          # [n_gt, n_pred]
        tp = match_predictions(pcls, gts[:, 4].astype(int), iou, IOUV)
    else:
        tp = np.zeros((len(pbox), len(IOUV)), dtype=bool)

    tp_l.append(tp); conf_l.append(pconf)
    pcls_l.append(pcls); tcls_l.append(gts[:, 4].astype(int) if len(gts) else np.zeros(0, dtype=int))

tp = np.concatenate(tp_l)
conf = np.concatenate(conf_l)
pred_cls = np.concatenate(pcls_l)
target_cls = np.concatenate(tcls_l)

n_gt = len(target_cls)
n_pred = len(conf)
print(f"GT objects: {n_gt}, predictions (conf>={0.001}): {n_pred}")

metrics = DetMetrics(names={0: "0"})
metrics.process(tp, conf, pred_cls, target_cls)
mp, mr, map50, map5095 = metrics.mean_results()

print("\n===== RKNN-fp16 on DAUB test (YOLO standard) =====")
print(f"Precision : {mp:.4f}")
print(f"Recall    : {mr:.4f}")
print(f"mAP@0.5   : {map50:.4f}")
print(f"mAP@0.5:0.95 : {map5095:.4f}")
