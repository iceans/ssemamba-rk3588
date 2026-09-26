"""轻量级多目标跟踪器：匀速运动模型 + 匈牙利匹配（用于金睛杯跟踪提交）"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class SimpleTracker:
    def __init__(self, dist_thr=25.0, max_missed=10):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.next_id = 1
        # 每条轨迹: {'id', 'cx', 'cy', 'vx', 'vy', 'missed'}
        self.tracks = []

    def update(self, dets):
        """dets: [(cx, cy), ...]，返回与 dets 对齐的 track id 列表"""
        if len(self.tracks) == 0:
            ids = []
            for (cx, cy) in dets:
                self.tracks.append({'id': self.next_id, 'cx': cx, 'cy': cy, 'vx': 0.0, 'vy': 0.0, 'missed': 0})
                ids.append(self.next_id)
                self.next_id += 1
            return ids

        n_tr, n_det = len(self.tracks), len(dets)
        ids = [None] * n_det

        if n_det > 0:
            # 预测位置 = 上一位置 + 速度
            BIG = 1e6
            cost = np.zeros((n_tr, n_det))
            for i, t in enumerate(self.tracks):
                px, py = t['cx'] + t['vx'], t['cy'] + t['vy']
                for j, (cx, cy) in enumerate(dets):
                    d = np.hypot(px - cx, py - cy)
                    cost[i, j] = d if d <= self.dist_thr else BIG
            row, col = linear_sum_assignment(cost)
            matched_tr = set()
            for i, j in zip(row, col):
                if cost[i, j] < BIG:  # 有效匹配
                    t = self.tracks[i]
                    cx, cy = dets[j]
                    t['vx'] = cx - t['cx']
                    t['vy'] = cy - t['cy']
                    t['cx'] = cx
                    t['cy'] = cy
                    t['missed'] = 0
                    ids[j] = t['id']
                    matched_tr.add(i)
            # 未匹配检测 -> 新轨迹
            for j in range(n_det):
                if ids[j] is None:
                    cx, cy = dets[j]
                    self.tracks.append({'id': self.next_id, 'cx': cx, 'cy': cy, 'vx': 0.0, 'vy': 0.0, 'missed': 0})
                    ids[j] = self.next_id
                    self.next_id += 1
            # 未匹配轨迹 -> 标记丢失
            for i in range(n_tr):
                if i not in matched_tr:
                    self.tracks[i]['missed'] += 1
        else:
            for t in self.tracks:
                t['missed'] += 1

        # 移除长期丢失的轨迹
        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]
        return ids
