"""
金睛杯 赛道一（红外视频卫星空中动目标检测）结果整理脚本
根据《参赛须知与提交结果说明》要求：
  - 每个序列输出一个 txt 文件，文件名 = 序列名（如 IR_00001.txt）
  - 每行：序列帧号 当前帧目标数 [目标编号 质心x 质心y]...
  - 仅检测任务：目标编号全部为 0
输入：tsgmamba cube 模型（10 通道 = 7 帧灰度 + 当前帧 RGB）
"""
import argparse
import glob
import os
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import torch

from ultralytics.utils import nms


def parse_seq_frame(path):
    """IR_00001_00007.bmp -> ('IR_00001', 7)"""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    parts = name.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1]), int(parts[-1])
    return name, 0


def build_cube(frames, idx, frame_num=7):
    """构造 10 通道输入：[BGR(当前帧) + 7 帧灰度]，再按 _format_img 反序并归一化"""
    win = frames[max(0, idx - frame_num + 1): idx + 1]
    while len(win) < frame_num:  # 序列起始处用首帧补齐
        win = [win[0]] + win
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)[..., None] for f in win]
    rgb = cv2.imread(win[-1])  # BGR, 当前帧
    im = np.concatenate([rgb] + imgs, axis=2)  # (H, W, 10)
    # 复现 augment.Format._format_img: transpose(2,0,1)[::-1], 再 /255
    im = np.ascontiguousarray(im.transpose(2, 0, 1)[::-1]).astype(np.float32) / 255.0
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/detect/train126/weights/best.pt')
    parser.add_argument('--img-dir', default='/home/dell/lxs/JinJingBei/SatVideoIRSDT_yolo/val/images')
    parser.add_argument('--out-dir', default='JJB_submit')
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--iou', type=float, default=0.65)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--frame-num', type=int, default=7)
    parser.add_argument('--zip', action='store_true', help='打包为 zip')
    args = parser.parse_args()

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = ck['model'].float().eval().cuda()

    all_imgs = sorted(glob.glob(os.path.join(args.img_dir, '*.bmp')))
    seqs = defaultdict(list)
    for p in all_imgs:
        seq, _ = parse_seq_frame(p)
        seqs[seq].append(p)
    seqs = {k: sorted(v, key=lambda x: parse_seq_frame(x)[1]) for k, v in sorted(seqs.items())}
    print(f'sequences: {len(seqs)}, images: {len(all_imgs)}')

    os.makedirs(args.out_dir, exist_ok=True)

    for seq, frames in seqs.items():
        # 逐帧构造 cube 输入
        xs = [build_cube(frames, i, args.frame_num) for i in range(len(frames))]
        dets_per_frame = []
        for s in range(0, len(xs), args.batch):
            batch = torch.from_numpy(np.stack(xs[s:s + args.batch])).cuda()
            with torch.no_grad():
                preds = model(batch)
            outs = nms.non_max_suppression(
                preds, args.conf, args.iou, nc=0, multi_label=True, agnostic=True, max_det=args.max_det
            )
            for o in outs:
                dets_per_frame.append(o.cpu().numpy() if o is not None and len(o) else np.zeros((0, 6)))

        out_path = os.path.join(args.out_dir, f'{seq}.txt')
        with open(out_path, 'w') as f:
            for frame_path, dets in zip(frames, dets_per_frame):
                _, fr = parse_seq_frame(frame_path)
                line = [str(fr), str(len(dets))]
                for d in dets:
                    cx = (d[0] + d[2]) / 2.0
                    cy = (d[1] + d[3]) / 2.0
                    line += ['0', f'{cx:.2f}', f'{cy:.2f}']
                f.write(' '.join(line) + '\n')
        print(f'  wrote {out_path} ({len(frames)} frames)')

    print('done ->', os.path.abspath(args.out_dir))

    if args.zip:
        zip_path = os.path.abspath(args.out_dir).rstrip('/') + '.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(args.out_dir)):
                if fn.endswith('.txt'):
                    zf.write(os.path.join(args.out_dir, fn), arcname=fn)
        print('zip ->', zip_path)


if __name__ == '__main__':
    main()
