"""
金睛杯 赛道一 正式测试集 检测+跟踪 结果整理脚本
输出格式（检测+跟踪）：序列帧号 当前帧目标数 [目标编号 质心x 质心y]...
  目标编号为跟踪 ID（区分不同目标），不再全为 0
"""
import argparse
import glob
import os
import zipfile

import cv2
import numpy as np
import torch

from ultralytics.utils import nms
from simple_tracker import SimpleTracker

IMG_EXTS = ('.jpg', '.jpeg', '.bmp', '.png')


def parse_frame_number(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    num = 0
    for t in stem.split('_'):
        if t.isdigit():
            num = int(t)
    return num


def build_cube_and_meta(frame_paths, idx, frame_num=7, imgsz=640):
    win = frame_paths[max(0, idx - frame_num + 1): idx + 1]
    while len(win) < frame_num:
        win = [win[0]] + win
    gray_imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in win]
    rgb = cv2.imread(win[-1])
    if rgb is None:
        raise FileNotFoundError(win[-1])
    h0, w0 = rgb.shape[:2]
    scale = imgsz / max(h0, w0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))

    def resize(img):
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    rgb = resize(rgb)
    gray_imgs = [resize(g)[..., None] for g in gray_imgs]
    im = np.concatenate([rgb] + gray_imgs, axis=2)
    dh, dw = imgsz - new_h, imgsz - new_w
    top, left = dh // 2, dw // 2
    im = np.pad(im, ((top, dh - top), (left, dw - left), (0, 0)), mode='constant', constant_values=114)
    im = np.ascontiguousarray(im.transpose(2, 0, 1)[::-1]).astype(np.float32) / 255.0
    return im, {'scale': scale, 'pad_left': left, 'pad_top': top}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/detect/train126/weights/best.pt')
    parser.add_argument('--img-root', default='/home/dell/lxs/JinJingBei/SatVideoIRSDT_v1_test/test/img')
    parser.add_argument('--out-dir', default='JJB_test_submit_track')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--iou', type=float, default=0.65)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--frame-num', type=int, default=7)
    parser.add_argument('--dist-thr', type=float, default=25.0)
    parser.add_argument('--max-missed', type=int, default=10)
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args()

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = ck['model'].float().eval().cuda()

    seq_dirs = sorted([d for d in glob.glob(os.path.join(args.img_root, '*')) if os.path.isdir(d)])
    print(f'sequences: {len(seq_dirs)}')

    os.makedirs(args.out_dir, exist_ok=True)

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        frames = []
        for ext in IMG_EXTS:
            frames += glob.glob(os.path.join(seq_dir, '*' + ext))
        frames = sorted(frames, key=parse_frame_number)

        # 逐帧推理，收集 (cx, cy) 原图坐标
        per_frame_dets = []
        for s in range(0, len(frames), args.batch):
            batch_frames = frames[s:s + args.batch]
            xs, metas = [], []
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
            for o, meta in zip(outs, metas):
                if o is None or len(o) == 0:
                    per_frame_dets.append([])
                    continue
                o = o.cpu().numpy()
                cur = []
                for d in o:
                    cx = ((d[0] - meta['pad_left']) / meta['scale'] + (d[2] - meta['pad_left']) / meta['scale']) / 2
                    cy = ((d[1] - meta['pad_top']) / meta['scale'] + (d[3] - meta['pad_top']) / meta['scale']) / 2
                    cur.append((cx, cy))
                per_frame_dets.append(cur)

        # 跟踪
        tracker = SimpleTracker(dist_thr=args.dist_thr, max_missed=args.max_missed)
        out_path = os.path.join(args.out_dir, f'{seq_name}.txt')
        with open(out_path, 'w') as f:
            for frame_path, dets in zip(frames, per_frame_dets):
                fr = parse_frame_number(frame_path)
                ids = tracker.update(dets)
                line = [str(fr), str(len(dets))]
                for tid, (cx, cy) in zip(ids, dets):
                    line += [str(tid), f'{cx:.2f}', f'{cy:.2f}']
                f.write(' '.join(line) + '\n')
        print(f'  wrote {out_path} ({len(frames)} frames)')

    print(f'done -> {os.path.abspath(args.out_dir)}')

    if args.zip:
        zip_path = os.path.abspath(args.out_dir).rstrip('/') + '.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(args.out_dir)):
                if fn.endswith('.txt'):
                    zf.write(os.path.join(args.out_dir, fn), arcname=fn)
        print('zip ->', zip_path)


if __name__ == '__main__':
    main()
