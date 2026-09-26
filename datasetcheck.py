# import os
# import cv2
# import torch
# import glob
# import shutil
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
#
# # [关键] 强制使用非交互式后端，防止 Ubuntu Qt 报错
# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from ultralytics.data.dataset import YOLODataset
#
# # ==========================================
# # 模块 1: 序列化数据集 (TSGSequenceDataset)
# # ==========================================
#
# class TSGSequenceDataset(Dataset):
#     def __init__(self, img_dir, label_dir, frame_num=5, img_size=(640, 640)):
#         self.img_dir = Path(img_dir)
#         self.label_dir = Path(label_dir)
#         self.frame_num = frame_num
#         self.img_size = img_size
#
#         # 1. 获取图片列表 (按文件名排序)
#         # 支持 .jpg 和 .png
#         self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png')))
#
#         if len(self.img_files) == 0:
#             raise ValueError(f"错误: 在 {img_dir} 未找到任何图片")
#
#         # 2. 生成合法索引 (防止跨视频采样)
#         self.valid_indices = self._make_valid_indices()
#         print(f"[Init] 加载图片 {len(self.img_files)} 张, 生成合法序列 {len(self.valid_indices)} 个")
#
#     def _make_valid_indices(self):
#         """
#         生成索引列表。此处为基础逻辑：截断最后 frame_num 帧。
#         [进阶] 如果文件名包含视频ID (e.g. video1_001.jpg)，请在此处添加 if id[i] == id[i+T] 判断
#         """
#         valid_indices = []
#         total = len(self.img_files)
#         # 简单截断，防止索引越界
#         for i in range(total - self.frame_num + 1):
#             valid_indices.append(i)
#         return valid_indices
#
#     def load_label(self, img_path):
#         """读取 YOLO 格式: class x_center y_center w h"""
#         label_name = img_path.stem + ".txt"
#         label_path = self.label_dir / label_name
#
#         boxes = []
#         if label_path.exists():
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     data = line.strip().split()
#                     if len(data) >= 5:
#                         boxes.append([float(x) for x in data])
#
#         if len(boxes) == 0:
#             return torch.zeros((0, 5))  # 空标签占位
#         return torch.tensor(boxes, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.valid_indices)
#
#     def __getitem__(self, index):
#         start_idx = self.valid_indices[index]
#
#         frames = []
#         labels = {}
#
#         # 连续读取 T 帧
#         for t in range(self.frame_num):
#             curr_idx = start_idx + t
#             img_path = self.img_files[curr_idx]
#
#             # --- 图像处理 (灰度模式) ---
#             img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # (H, W)
#             if img is None:
#                 raise ValueError(f"无法读取图片: {img_path}")
#
#             # Resize
#             img = cv2.resize(img, self.img_size)
#
#             # 归一化 [0, 1]
#             img = img.astype(np.float32) / 255.0
#             frames.append(img)
#
#             # --- 标签处理 ---
#             # 读取并存入对应 Key: bboxes_t1 ... bboxes_t5
#             labels[f'bboxes_t{t + 1}'] = self.load_label(img_path)
#
#         # 堆叠为 (T, H, W) -> 注意：没有 Channel 维度
#         seq_tensor = np.stack(frames, axis=0)
#
#         data = {
#             'img': torch.from_numpy(seq_tensor).float(),
#             'img_path': str(self.img_files[start_idx])
#         }
#         data.update(labels)
#
#         return data
#
#
# # ==========================================
# # 模块 2: Collate Function (处理变长 Label)
# # ==========================================
# def tsg_collate_fn(batch):
#     batch_out = {}
#
#     # 1. 图像 Stack: [B, T, H, W]
#     imgs = [item['img'] for item in batch]
#     batch_out['img'] = torch.stack(imgs, axis=0)
#
#     # 2. 标签保持 List 结构: key -> List[Tensor] (len=Batch)
#     # 自动查找所有 'bboxes_t' 开头的键
#     keys = [k for k in batch[0].keys() if 'bboxes' in k]
#     for k in keys:
#         batch_out[k] = [item[k] for item in batch]
#
#     # 3. 路径信息 (可选)
#     batch_out['img_path'] = [item['img_path'] for item in batch]
#
#     return batch_out
#
#
# # ==========================================
# # 模块 3: 高清可视化验证器 (HighResValidator)
# # ==========================================
# class DatasetValidator:
#     def __init__(self, output_dir="check_vis_hd"):
#         self.output_dir = Path(output_dir)
#         # 每次运行前清空目录，避免混淆
#         if self.output_dir.exists():
#             shutil.rmtree(self.output_dir)
#         self.output_dir.mkdir(parents=True)
#         print(f"[Vis] 检查图像将保存至: {self.output_dir.resolve()}")
#
#     def denormalize_box(self, box, img_w, img_h):
#         """YOLO (cx, cy, w, h) -> Pixel (x1, y1, x2, y2)"""
#         xc, yc, w, h = box[0], box[1], box[2], box[3]
#         x1 = int((xc - w / 2) * img_w)
#         y1 = int((yc - h / 2) * img_h)
#         x2 = int((xc + w / 2) * img_w)
#         y2 = int((yc + h / 2) * img_h)
#         return int(1), x1, y1, x2, y2
#
#     def visualize(self, batch, batch_idx=0):
#         """
#         核心逻辑：每个样本生成一张独立的超宽高清图
#         """
#         images = batch['img']  # [B, T, H, W]
#         if isinstance(images, torch.Tensor):
#             images = images.numpy()
#
#         # 维度兼容性检查
#         if images.ndim == 5:
#             images = images.squeeze(2)  # 如果混入了 C=1，压缩掉
#
#         B, T, H, W = images.shape
#
#         # 遍历 Batch 中的每一个样本
#         for b in range(B):
#             # 1. 创建高清画布
#             # 宽 = 帧数 * 6 英寸 (约 30 英寸), 高 = 6 英寸
#             # DPI = 150 -> 最终宽度约 4500px, 高度 900px
#             fig, axes = plt.subplots(1, T, figsize=(T * 6, 6))
#             if T == 1: axes = [axes]
#
#             for t in range(T):
#                 # --- A. 图像恢复 ---
#                 # Float [0,1] -> Uint8 [0,255]
#                 img_gray = (images[b, t] * 255).astype(np.uint8)
#                 img_gray = np.ascontiguousarray(img_gray)
#
#                 # [关键] 灰度转 BGR，否则画不出红色框
#                 img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
#
#                 # --- B. 绘制标签 ---
#                 key = f'bboxes_t{t + 1}'
#                 if key in batch:
#                     boxes = batch[key][b]  # 获取 Tensor [N, 5]
#
#                     for box in boxes:
#                         # 过滤 Padding (全0行)
#                         if torch.sum(box) == 0: continue
#
#                         cls, x1, y1, x2, y2 = self.denormalize_box(box, W, H)
#
#                         # [优化] 线条加粗 (Thickness=3), 字体加大 (Scale=1.5)
#                         cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
#                         # 标签背景
#                         label_text = f"C{cls}"
#                         (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
#                         cv2.rectangle(img_vis, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
#
#                         # 标签文字
#                         cv2.putText(img_vis, label_text, (x1 + 5, y1 - 8),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
#
#                 # --- C. Matplotlib 展示 ---
#                 ax = axes[t]
#                 # OpenCV BGR -> Matplotlib RGB
#                 ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
#                 ax.set_title(f"Frame {t + 1}", fontsize=18, fontweight='bold')
#                 ax.axis('off')
#
#             # --- D. 保存 ---
#             # 去除多余白边
#             plt.tight_layout()
#
#             save_name = f"batch_{batch_idx}_sample_{b}.jpg"
#             save_path = self.output_dir / save_name
#
#             # 使用高 DPI 保存
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             plt.close(fig)  # 释放内存
#
#             print(f" > [OK] 保存高清样本: {save_path}")
#
#
# # ==========================================
# # 主程序入口 (Main)
# # ==========================================
# if __name__ == "__main__":
#     # --- 配置路径 ---
#     # 请替换为您服务器上的真实路径
#     # 例如: "/home/dell/data/coco/images/train"
#     # IMG_PATH = "/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/images"
#     # LBL_PATH = "/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/labels"
#     IMG_PATH = "/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/images"
#     LBL_PATH = "/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/labels"
#
#     # 检查路径是否存在 (仅作提示)
#     if not os.path.exists(IMG_PATH):
#         print(f"[Warning] 路径 {IMG_PATH} 不存在，请修改代码中的 IMG_PATH")
#         # 为了演示，创建一个假文件防止报错 (仅供复制运行测试)
#         os.makedirs(IMG_PATH, exist_ok=True)
#         os.makedirs(LBL_PATH, exist_ok=True)
#         # 创建假图片
#         dummy = np.zeros((640, 640), dtype=np.uint8)
#         cv2.imwrite(f"{IMG_PATH}/0001.jpg", dummy)
#         cv2.imwrite(f"{IMG_PATH}/0002.jpg", dummy)
#         cv2.imwrite(f"{IMG_PATH}/0003.jpg", dummy)
#         cv2.imwrite(f"{IMG_PATH}/0004.jpg", dummy)
#         cv2.imwrite(f"{IMG_PATH}/0005.jpg", dummy)
#         # 创建假标签
#         with open(f"{LBL_PATH}/0001.txt", "w") as f:
#             f.write("0 0.5 0.5 0.2 0.2\n")  # Center box
#
#     try:
#         # 1. 初始化数据集
#         from types import SimpleNamespace
#         import yaml
#         yaml_path = "/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/ITSDT.yaml"
#         with open(yaml_path,'r') as f:
#             data_cfg = yaml.safe_load(f)
#         hyp = SimpleNamespace(
#             mosaic=1.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.1, scale=0.5,
#             shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#             mask_ratio=1, overlap_mask=True,
#             # data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/UAVSwarm.yaml", batch=128, lr0=0.04,
#             # epochs=50,
#             # device=[0, 1], patience=20, save_json=False, resume=False,
#             augment = False,
#             cube=True,
#             dwa=False, val=True, gray=True, mod=False, frame_num=5,
#         )
#         # hyp={'cube':True,'frame_num':5,}
#         dataset = YOLODataset(IMG_PATH, imgsz=640, hyp =hyp,data = data_cfg)
#
#         # 2. 初始化加载器
#         loader = DataLoader(
#             dataset,
#             batch_size=1,  # 即使 Batch 很大，验证器也会拆分保存
#             shuffle=False,
#             collate_fn=tsg_collate_fn,  # 必须使用自定义 Collate
#             num_workers=4,
#
#         )
#
#         # 3. 初始化验证器
#         validator = DatasetValidator(output_dir="check_vis_hd")
#
#         print("\n=== 开始 Pipeline 检查 ===")
#         # for i, batch in enumerate(loader):
#         #     # 打印形状信息
#         #     if i == 0:
#         #         print(f"Batch Tensor Shape: {batch['img'].shape} (期望: [B, 5, 640, 640])")
#         #
#         #     # 执行可视化
#         #     validator.visualize(batch, batch_idx=i)
#         #
#         #     # 仅检查前 2 个 Batch，避免生成太多图片
#         #     if i >= 1:
#         #         break
#         idx = 4
#         data = dataset[idx]
#         print(data['im_file'])
#         batch = {
#             'img': data['img'].unsqueeze(0), # [1, T, H, W]
#             'bboxes_t1': [data['bboxes_t1']], # List [Tensor]
#             'bboxes_t2': [data['bboxes_t2']],
#             'bboxes_t3': [data['bboxes_t3']],
#             'bboxes_t4': [data['bboxes_t4']],
#             'bboxes_t5': [data['bboxes_t5']],}
#         validator.visualize(batch, batch_idx=idx)
#         print("\n=== 检查完成 ===")
#         print(f"请查看文件夹: {os.path.abspath('check_vis_hd')}")
#
#     except Exception as e:
#         print(f"\n[Error] 发生错误: {e}")
#         import traceback
#
#         traceback.print_exc()
import os
import cv2
import torch
import glob
import shutil
import numpy as np
import yaml
import sys
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # [新增] 用于显示进度条

# [关键] 强制使用非交互式后端，防止 Ubuntu Qt 报错
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 假设你的 YOLODataset 位于此处，如果路径不对请修改
from ultralytics.data.dataset import YOLODataset


# ==========================================
# 模块 1: Collate Function (处理变长 Label)
# ==========================================
def tsg_collate_fn(batch):
    batch_out = {}
    # 1. 图像 Stack: [B, T, H, W] (如果 Dataset 返回的是 T,H,W)
    # 注意：ultralytics YOLODataset 通常返回的 img 是 (C, H, W) 或 (T*C, H, W)
    # 这里我们假设你的 dataset 修改后返回的是符合 cube 逻辑的 tensor
    imgs = [item['img'] for item in batch]
    batch_out['img'] = torch.stack(imgs, axis=0)

    # 2. 标签保持 List 结构
    # 自动查找所有 'bboxes' 开头的键 (bboxes_t1, bboxes_t2...)
    keys = [k for k in batch[0].keys() if 'bboxes' in k]
    for k in keys:
        batch_out[k] = [item[k] for item in batch]

    # 3. 路径信息
    if 'im_file' in batch[0]:
        batch_out['im_file'] = [item['im_file'] for item in batch]

    return batch_out


# ==========================================
# 模块 2: 高清可视化验证器
# ==========================================
class DatasetValidator:
    def __init__(self, output_dir="check_vis_hd"):
        self.output_dir = Path(output_dir)
        # 每次运行前清空目录，避免混淆
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        print(f"[Vis] 检查结果将保存至: {self.output_dir.resolve()}")

    def denormalize_box(self, box, img_w, img_h):
        """YOLO (cls, cx, cy, w, h) -> Pixel (cls, x1, y1, x2, y2)"""
        # [修复] 原始代码这里是 box[0]... 但 box 通常包含 cls
        # 假设 box = [cls, xc, yc, w, h]
        # cls_id = int(box[0])
        xc, yc, w, h = box[0], box[1], box[2], box[3]

        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        return  x1, y1, x2, y2

    def visualize(self, batch, batch_idx=0):
        images = batch['img']
        if isinstance(images, torch.Tensor):
            images = images.numpy()

        # 维度处理: [B, T, C, H, W] -> [B, T, H, W]
        # 如果你的 Dataset 输出是 [B, 5, 640, 640] (无 Channel)，这步会跳过
        # 如果输出是 [B, 15, 640, 640] (Stack Channel)，需要手动 split，这里假设是 [B, T, H, W]
        if images.ndim == 5 and images.shape[2] == 1:
            images = images.squeeze(2)

        B, T, H, W = images.shape

        # 遍历 Batch 中的每一个样本
        for b in range(B):
            # 创建高清画布: 宽 = 帧数 * 5 英寸
            fig, axes = plt.subplots(1, T, figsize=(T * 5, 5))
            if T == 1: axes = [axes]

            for t in range(T):
                # --- A. 图像恢复 ---
                # 假设输入是归一化的 [0,1]
                img_gray = (images[b, t] * 255).astype(np.uint8)
                img_gray = np.ascontiguousarray(img_gray)

                # 转为 BGR 以便画彩色框
                img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

                # --- B. 绘制标签 ---
                # 你的 Dataset 输出 key 为 bboxes_t1, bboxes_t2...
                key = f'bboxes_t{t + 1}'
                if key in batch:
                    boxes = batch[key][b]  # List[Tensor] -> Tensor

                    for box in boxes:
                        # 过滤 Padding (全0行)
                        if torch.sum(box) == 0: continue
                        x1, y1, x2, y2 = self.denormalize_box(box, W, H)

                        # 绘制
                        color = (0, 0, 255)  # Red
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

                        # 文字
                        label_text = f"C{1}"
                        cv2.putText(img_vis, label_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # --- C. Matplotlib 展示 ---
                ax = axes[t]
                ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))

                # 获取文件名信息用于 Title (如果 collate_fn 收集了)
                title_suffix = ""
                if 'im_file' in batch:
                    fname = os.path.basename(batch['im_file'][b])
                    title_suffix = f"\n{fname}"

                ax.set_title(f"T{t + 1}{title_suffix}", fontsize=10)
                ax.axis('off')

            # --- D. 保存 ---
            plt.tight_layout()
            # 命名规则: batch_索引_样本_索引.jpg
            save_name = f"batch_{batch_idx:04d}_sample_{b}.jpg"
            save_path = self.output_dir / save_name

            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)  # [关键] 必须关闭，否则内存爆炸


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 配置路径
    # IMG_PATH = "/home/dell/lxs/Anti_UAV_dataset/ITSDT-15K/yolo_format/images/val"
    # LBL_PATH = "/home/dell/lxs/Anti_UAV_dataset/ITSDT-15K/yolo_format/labels/val"
    # IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/STtran/images/anti_test'
    # LBL_PATH ='/home/dell/lxs/Anti_UAV_dataset/STtran/labels/anti_test'
    IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/images'
    LBL_PATH ='/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/labels'

    # YAML 配置
    yaml_path = "/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/UAVSwarm.yaml"

    try:
        print("=== 初始化配置 ===")
        # 加载 YAML 数据字典
        with open(yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)

        # 定义超参数
        # hyp = SimpleNamespace(
        #     mosaic=0.0, mixup=0.0,  # 验证模式建议关闭增强
        #     augment=False,
        #     cube=True,
        #     val=True,
        #     gray=True,
        #     frame_num=5,
        #     imgsz=640
        # )
        # 定义超参数 (务必包含所有 YOLOv8 所需的 key)
        hyp = SimpleNamespace(
            mosaic=0.0,
            mixup=0.0,

            # --- [关键修复] 添加 copy_paste ---
            copy_paste=0.0,  # 验证时设为 0.0 关闭

            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mask_ratio=1,
            overlap_mask=True,

            # 自定义参数
            augment=True,
            cube=True,
            val=True,
            gray=True,
            frame_num=3,
            imgsz=640
        )
        # 2. 初始化 Dataset
        # 注意：这里使用你库里的 YOLODataset，请确保该类已修改支持 cube/frame_num
        dataset = YOLODataset(
            img_path=IMG_PATH,
            imgsz=640,
            hyp=hyp,
            data=data_cfg
        )
        print(f"Dataset 长度: {len(dataset)}")

        # 3. 初始化 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=1,  # 建议设为 1，这样文件名和图像一一对应，方便查错
            shuffle=False,  # 顺序读取，不打乱
            collate_fn=tsg_collate_fn,
            num_workers=4
        )

        # 4. 初始化验证器
        validator = DatasetValidator(output_dir="check_vis_hd")

        # 5. [核心] 遍历所有数据
        print("\n=== 开始全量数据检查 (按 Ctrl+C 可中断) ===")

        # 使用 tqdm 显示进度条
        for i, batch in enumerate(tqdm(loader, total=len(loader), unit="img")):
            try:
                if i == 1498:
                    print('1')
                validator.visualize(batch, batch_idx=i)

            except Exception as e_vis:
                print(f"[Warn] Batch {i} 可视化失败: {e_vis}")
                continue

        print("\n=== 检查完成 ===")
        print(f"结果保存在: {os.path.abspath('check_vis_hd')}")

    except KeyboardInterrupt:
        print("\n[Info] 用户手动中断检查。")
    except Exception as e:
        print(f"\n[Error] 发生严重错误: {e}")
        import traceback

        traceback.print_exc()