import torch
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from types import SimpleNamespace
from ultralytics.data.dataset import YOLODataset


# ----------------------------------------------------------------------
# 配置部分 (保持不变)
# ----------------------------------------------------------------------
def get_config():
    data_yaml_path = '/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/UAVSwarm.yaml'
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    hyp = SimpleNamespace(
        mosaic=1.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mask_ratio=1, overlap_mask=True,
        data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/UAVSwarm.yaml", batch=128, lr0=0.04, epochs=50,
        device=[0, 1], patience=20, save_json=False, resume=False, cube=True,
        dwa=False, val=True, gray=True, mod=False,frame_num=5
    )
    return data_cfg, hyp


# ----------------------------------------------------------------------
# 核心验证逻辑 (修复了针对 Tuple 的报错)
# ----------------------------------------------------------------------
def validate_dataset():
    data_cfg, hyp = get_config()
    # img_path = '/home/dell/lxs/Anti_UAV_dataset/STtran/images/anti_test'
    img_path = '/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test'
    print(f"Loading data from: {img_path}")

    # 1. 实例化 Dataset
    dataset = YOLODataset(
        img_path=img_path, imgsz=640, batch_size=8, augment=True, hyp=hyp,
        rect=False, cache=False, single_cls=False, stride=32, pad=0.0, data=data_cfg
    )
    print(f"Dataset loaded. Length: {len(dataset)}")

    # 2. 实例化 DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=YOLODataset.collate_fn)

    # 3. 取出一个 Batch
    print("Fetching a batch...")
    # try:
    batch = next(iter(loader))
    # except Exception as e:
    #     print(f"❌ Error fetching batch: {e}")
    #     return

    imgs = batch['img']
    print(f"\n[Check A] Image Tensor Shape: {imgs.shape}")

    # ------------------------------------------------------------------
    # 可视化部分 (Robust Version)
    # ------------------------------------------------------------------
    print("\n[Check C] Visualizing first sample...")
    target_idx = 0  # 我们只看 batch 中的第 0 张图

    # 图片反归一化
    img_tensor = imgs[target_idx]
    img_numpy = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0
    img_numpy = np.ascontiguousarray(img_numpy, dtype=np.uint8)

    # 准备画布
    num_channels = img_numpy.shape[2]
    # 如果通道太多(>9)，只画前4个，防止报错
    plot_channels = min(num_channels, 4)

    fig, axes = plt.subplots(1, plot_channels, figsize=(5 * plot_channels, 5))
    if plot_channels == 1: axes = [axes]

    frames = ['t1', 't2', 't3', 't4']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i in range(plot_channels):
        current_img_gray = img_numpy[:, :, i]
        current_img_vis = cv2.cvtColor(current_img_gray, cv2.COLOR_GRAY2RGB)

        # 确定 Key
        bbox_key = f'bboxes_{frames[i]}' if i < len(frames) else None

        if bbox_key in batch:
            data = batch[bbox_key]
            bboxes = None

            # [关键修复]：分类讨论 data 的类型
            print(f"   -> Processing {bbox_key}, Type: {type(data)}")

            # 情况 A: 如果是 Tuple 或 List (说明每张图的 Tensor 被放在列表里了)
            if isinstance(data, (tuple, list)):
                # 检查索引是否越界
                if target_idx < len(data):
                    bboxes = data[target_idx]
                else:
                    print(f"      ⚠️ Index {target_idx} out of range for tuple len {len(data)}")

            # 情况 B: 如果是 Tensor (说明堆叠在一起了)
            elif isinstance(data, torch.Tensor):
                # 子情况 B1: [Batch, 4] -> 固定每个样本1个框
                if data.shape[0] == 4 and data.ndim == 2:  # 假设 Batch=4
                    bboxes = data[target_idx].unsqueeze(0)
                # 子情况 B2: [Batch, N, 4] -> 每个样本N个框
                elif data.ndim == 3:
                    bboxes = data[target_idx]
                # 子情况 B3: [Total_Boxes, 6] -> YOLO 格式 (batch_idx, cls, x, y, w, h)
                elif 'batch_idx' in batch:
                    idx_mask = batch['batch_idx'] == target_idx
                    bboxes = data[idx_mask]

            # 统一画框逻辑
            if bboxes is not None and isinstance(bboxes, torch.Tensor):
                h, w, _ = current_img_vis.shape
                for box in bboxes:
                    # 无论 box 是一维还是二维，转为列表
                    b_list = box.view(-1).tolist()
                    if len(b_list) >= 4:
                        cx, cy, bw, bh = b_list[-4:]  # 取最后4个作为坐标
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(current_img_vis, (x1, y1), (x2, y2), colors[i % len(colors)], 1)

        axes[i].imshow(current_img_vis)
        axes[i].set_title(f"Ch {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_validation_v2.jpg')
    print("✅ Visualization saved to 'dataset_validation_v2.jpg'")


if __name__ == '__main__':
    validate_dataset()