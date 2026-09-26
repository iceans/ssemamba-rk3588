import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def seed_worker(worker_id):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


@lru_cache(maxsize=128)
def image_bgr(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(file)
    return image


def letterbox(image, boxes, imgsz):
    h, w = image.shape[:2]
    gain = imgsz / max(h, w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (imgsz - nw) // 2, (imgsz - nh) // 2
    output = np.full((imgsz, imgsz, image.shape[2]), 114, dtype=np.uint8)
    output[top:top + nh, left:left + nw] = resized
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * gain + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * gain + top
    return output, boxes, ((gain, gain), (left, top))


class ClipDataset(Dataset):
    """One causal clip per manifest keyframe; annotations belong only to that keyframe."""
    def __init__(self, manifest, imgsz=640, train=False, seed=0, epoch=0,
                 temporal_stride=1, common_only=False, limit=None):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        self.imgsz, self.train, self.seed, self.epoch = imgsz, train, seed, epoch
        self.temporal_stride = temporal_stride
        self.sequences = defaultdict(list)
        for i, row in enumerate(self.rows):
            self.sequences[row['seq_id']].append(i)
        self.position = {}
        for seq, indices in self.sequences.items():
            indices.sort(key=lambda i: self.rows[i]['frame_id'])
            for pos, i in enumerate(indices):
                self.position[i] = pos
        self.indices = [i for i in range(len(self.rows)) if not common_only or self.position[i] >= 16]
        if limit is not None:
            self.indices = self.indices[:limit]
        self.labels = [{'cls': np.array([[b['class']] for b in self.rows[i]['boxes']], dtype=np.float32)}
                       for i in self.indices]
        assert len({self.rows[i]['file'] for i in self.indices}) == len(self.indices)
        self.rect = False

    def __len__(self):
        return len(self.indices)

    def clip_indices(self, index):
        absolute = self.indices[index]
        row = self.rows[absolute]
        pos = self.position[absolute]
        sequence = self.sequences[row['seq_id']]
        return [sequence[max(0, pos - j * self.temporal_stride)] for j in range(4, -1, -1)]

    def __getitem__(self, index):
        absolute = self.indices[index]
        row = self.rows[absolute]
        clip = self.clip_indices(index)
        assert clip[-1] == absolute
        assert all(self.rows[i]['seq_id'] == row['seq_id'] for i in clip)
        key = image_bgr(row['file'])
        assert key.shape[:2] == (row['height'], row['width'])
        temporal = [cv2.cvtColor(image_bgr(self.rows[i]['file']), cv2.COLOR_BGR2GRAY) for i in clip]
        assert all(x.shape == key.shape[:2] for x in temporal)
        image = np.concatenate((np.stack(temporal, axis=2), key[:, :, ::-1]), axis=2)
        boxes = np.array([b['xyxy'] for b in row['boxes']], dtype=np.float32).reshape(-1, 4)
        image, boxes, ratio_pad = letterbox(image, boxes, self.imgsz)
        # The audited environment's active geometric augmentation is fliplr=0.5.
        # Stateless per-example sampling keeps clips and labels aligned across workers/resume.
        rng = random.Random(self.seed * 1000000007 + self.epoch * 1000003 + absolute)
        flipped = self.train and rng.random() < .5
        if flipped:
            image = image[:, ::-1]
            boxes[:, [0, 2]] = self.imgsz - boxes[:, [2, 0]]
        xywh = boxes.copy()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        xywh /= self.imgsz
        cls = torch.tensor([[b['class']] for b in row['boxes']], dtype=torch.float32).reshape(-1, 1)
        return {
            'img': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            'cls': cls, 'bboxes_t5': torch.from_numpy(xywh),
            'batch_idx': torch.zeros(len(cls)), 'im_file': row['file'],
            'ori_shape': (row['height'], row['width']), 'ratio_pad': ratio_pad,
            'seq_id': row['seq_id'], 'frame_id': row['frame_id'],
            'clip_files': [self.rows[i]['file'] for i in clip], 'flipped': bool(flipped),
        }


def collate(batch):
    result = {}
    for key in batch[0]:
        if key == 'img':
            result[key] = torch.stack([r[key] for r in batch])
        elif key in ['cls', 'bboxes_t5']:
            result[key] = torch.cat([r[key] for r in batch])
        elif key == 'batch_idx':
            result[key] = torch.cat([r[key] + i for i, r in enumerate(batch)])
        else:
            result[key] = [r[key] for r in batch]
    return result


def loader(dataset, batch=8, workers=8):
    generator = torch.Generator().manual_seed(dataset.seed + 1000003 * dataset.epoch)
    return DataLoader(dataset, batch_size=batch, shuffle=dataset.train,
                      num_workers=workers, pin_memory=True, drop_last=False,
                      collate_fn=collate, generator=generator, worker_init_fn=seed_worker)


def to_device(batch, device):
    result = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    result['img'] = result['img'].float() / 255.
    return result
