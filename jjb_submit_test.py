"""
金睛杯 赛道一 正式测试集结果整理脚本
测试集结构：test/img/<序列目录>/<帧图片>
  - 每个序列目录输出一个 txt，文件名 = 序列目录名
  - 每行：序列帧号 当前帧目标数 [目标编号 质心x 质心y]...
  - 检测任务目标编号全为 0
模型：tsgmamba cube（10 通道 = 7 帧灰度 + 当前帧 RGB），输入 640x640
"""
import argparse
import glob
import os
import re
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import torch

from ultralytics.utils import nms

IMG_EXTS = ('.jpg', '.jpeg', '.bmp', '.png')


def parse_frame_number(path):
    """从文件名提取帧号：取下划线分割后最后一个纯数字 token（兼容 000001.jpg / 002500.bmp / CX..._01049_Raw_L0.png）"""
    stem = os.path.splitext(os.path.basename(path))[0]
    tokens = stem.split('_')
    num = 0
    for t in tokens:
        if t.isdigit():
            num = int(t)
    return num


def build_cube_and_meta(frame_paths, idx, frame_num=7, imgsz=640):
    """构造 10 通道 cube，并返回缩放/填充参数用于坐标还原。"""
    # 序列起始处用首帧补齐
    win = frame_paths[max(0, idx - frame_num + 1): idx + 1]
    while len(win) < frame_num:
        win = [win[0]] + win

    gray_imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in win]
    rgb = cv2.imread(win[-1])  # BGR 当前帧
    if rgb is None:
        raise FileNotFoundError(win[-1])

    h0, w0 = rgb.shape[:2]
    # 复现 load_image: 长边缩放到 imgsz
    scale = imgsz / max(h0, w0)
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))

    def resize(img):
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    rgb = resize(rgb)
    gray_imgs = [resize(g)[..., None] for g in gray_imgs]
    im = np.concatenate([rgb] + gray_imgs, axis=2)  # (H, W, 10)

    # 复现 LetterBox: 填充到 imgsz x imgsz（114 灰边）
    dh = imgsz - new_h
    dw = imgsz - new_w
    top = dh // 2
    left = dw // 2
    im = np.pad(im, ((top, dh - top), (left, dw - left), (0, 0)), mode='constant', constant_values=114)

    # 复现 Format._format_img: transpose(2,0,1)[::-1], /255
    im = np.ascontiguousarray(im.transpose(2, 0, 1)[::-1]).astype(np.float32) / 255.0
    meta = {'scale': scale, 'pad_left': left, 'pad_top': top, 'orig_w': w0, 'orig_h': h0}
    return im, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/detect/train126/weights/best.pt')
    parser.add_argument('--img-root', default='/home/dell/lxs/JinJingBei/SatVideoIRSDT_v1_test/test/img')
    parser.add_argument('--out-dir', default='JJB_test_submit')
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--iou', type=float, default=0.65)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--frame-num', type=int, default=7)
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args()

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = ck['model'].float().eval().cuda()

    seq_dirs = sorted([d for d in glob.glob(os.path.join(args.img_root, '*')) if os.path.isdir(d)])
    print(f'sequences: {len(seq_dirs)}')

    os.makedirs(args.out_dir, exist_ok=True)
    total_frames = 0

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        frames = []
        for ext in IMG_EXTS:
            frames += glob.glob(os.path.join(seq_dir, '*' + ext))
        # 按帧号排序
        frames = sorted(frames, key=parse_frame_number)

        # 逐帧推理
        dets_per_frame = []
        metas = []
        for s in range(0, len(frames), args.batch):
            batch_frames = frames[s:s + args.batch]
            xs = []
            for j in range(len(batch_frames)):
                x, meta = build_cube_and_meta(frames, s + j, args.frame_num)
                xs.append(x)
                metas.append(meta)
            batch = torch.from_numpy(np.stack(xs)).cuda()
            with torch.no_grad():
                preds = model(batch)
            outs = nms.non_max_suppression(
                preds, args.conf, args.iou, nc=0, multi_label=True, agnostic=True, max_det=args.max_det
            )
            for o in outs:
                dets_per_frame.append(o.cpu().numpy() if o is not None and len(o) else np.zeros((0, 6)))

        out_path = os.path.join(args.out_dir, f'{seq_name}.txt')
        with open(out_path, 'w') as f:
            for frame_path, dets, meta in zip(frames, dets_per_frame, metas):
                fr = parse_frame_number(frame_path)
                line = [str(fr), str(len(dets))]
                for d in dets:
                    # 640 空间 -> 原图坐标
                    x1 = (d[0] - meta['pad_left']) / meta['scale']
                    y1 = (d[1] - meta['pad_top']) / meta['scale']
                    x2 = (d[2] - meta['pad_left']) / meta['scale']
                    y2 = (d[3] - meta['pad_top']) / meta['scale']
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    line += ['0', f'{cx:.2f}', f'{cy:.2f}']
                f.write(' '.join(line) + '\n')
        total_frames += len(frames)
        print(f'  wrote {out_path} ({len(frames)} frames, frame range {parse_frame_number(frames[0])}..{parse_frame_number(frames[-1])})')

    print(f'done -> {os.path.abspath(args.out_dir)}, total frames: {total_frames}')

    if args.zip:
        zip_path = os.path.abspath(args.out_dir).rstrip('/') + '.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(args.out_dir)):
                if fn.endswith('.txt'):
                    zf.write(os.path.join(args.out_dir, fn), arcname=fn)
        print('zip ->', zip_path)


if __name__ == '__main__':
    main()
