# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS
from .utils import HELP_URL, IMG_FORMATS
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

import torch
import cv2
import numpy as np
import os
from pathlib import Path
import os
import cv2
import torch
import numpy as np
from pathlib import Path


def visualize_and_save(data_dict, save_dir="./visualization_results"):
    """
    针对 6 通道图像和 Instances 对象的自动保存脚本
    """
    # 1. 自动构建保存路径
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 2. 图像预处理 (取前三通道)
    # 假设 img 为 tensor [C, H, W] 或 numpy [C, H, W]
    img_data = data_dict['img']
    if isinstance(img_data, torch.Tensor):
        img_data = img_data.cpu().numpy()

    # 提取前 3 通道 (RGB) 并转换为 OpenCV 的 HWC BGR 格式
    # 如果 img 是 [6, H, W]，则取 img_data[:3, :, :]
    color_img = img_data[:, :, :3].astype(np.uint8)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    h, w = color_img.shape[:2]

    # 3. 解析 instances_t3 标签
    # Ultralytics Instances 对象通常包含 .bboxes (归一化或像素坐标)
    # 根据你的描述，这里手动按 xywh 归一化处理
    instances = data_dict['instances_t3']

    # 尝试从 Instances 对象中提取 bboxes
    # 注：不同版本的 Ultralytics 接口略有差异，通常使用 .bboxes 获取
    if hasattr(instances, 'bboxes'):
        bboxes = instances.bboxes  # 如果是 Tensor，需转为 numpy
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
    else:
        # 如果是其他自定义对象，请确认提取方式
        print("Warning: Could not find 'bboxes' attribute in instances_t3.")
        bboxes = []

    # 4. 绘制边界框
    for box in bboxes:
        # xywh (归一化格式)
        x_c, y_c, bw, bh = box

        # 转换为像素坐标 (Top-left x, y, Bottom-right x, y)
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # 绘制绿色矩形框
        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 5. 获取文件名并保存
    img_name = Path(data_dict['im_file']).name
    output_filename = str(save_path / f"vis_{img_name}")

    cv2.imwrite(output_filename, color_img)
    print(f"Successfully saved visualized image to: {output_filename}")


# --- 执行示例 ---
# 假设你的数据字典名为 batch_data
# visualize_and_save(batch_data)
def save_visualized_batch(data_dict, output_dir="./visualization_results"):
    """
    可视化并将结果保存到指定路径
    :param data_dict: 包含 'img', 'bboxes_t3', 'im_file' 等键的字典
    :param output_dir: 保存图片的目录
    """
    # 1. 自动构建路径
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. 提取图像数据 (前3通道为RGB)
    # img shape: [C, H, W] -> [6, 338, 640]
    img_tensor = data_dict['img']
    color_img = img_tensor[-3:, :, :].permute(1, 2, 0).numpy()

    # 转换 RGB 为 BGR (OpenCV 格式)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    h, w = color_img.shape[:2]

    # 3. 提取边界框 (bboxes_t3: xywh 归一化格式)
    if data_dict['bboxes_t3'] is not None:
        bboxes = data_dict['bboxes_t3']
    else:
        bboxes = data_dict['instances_t3'].bboxes
    for box in bboxes:
        x_c, y_c, bw, bh = box.tolist()

        # 还原坐标到像素尺度
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # 绘制矩形框 (绿色, 粗细为2)
        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. 获取原始文件名并保存
    origin_name = Path(data_dict['im_file']).name
    save_file_path = output_path / f"vis_{origin_name}"

    cv2.imwrite(str(save_file_path), color_img)
    print(f"Result saved to: {save_file_path}")
class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.cube = hyp.cube
        self.gray = hyp.gray
        # frame set
        self.frame_num = hyp.frame_num

        self.im_files = self.get_img_files(self.img_path)
        # self.frame_num = 4
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache stuff
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)


    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                if self.cube:
                    bboxes_t1 = self.labels[i]['bboxes_t1']
                    bboxes_t2 = self.labels[i]['bboxes_t2']
                    bboxes_t3 = self.labels[i]['bboxes_t3']
                    # bboxes_t4 = self.labels[i]['bboxes_t4']
                    # bboxes_t5 = self.labels[i]['bboxes_t5']
                    # bboxes_t6 = self.labels[i]['bboxes_t6']
                    segments = self.labels[i]['segments']
                    keypoints = self.labels[i]['keypoints']
                    j = (cls == include_class_array).any(1)
                    self.labels[i]['cls'] = cls[j]
                    self.labels[i]['bboxes_t1'] = bboxes_t1[j]
                    self.labels[i]['bboxes_t2'] = bboxes_t2[j]
                    self.labels[i]['bboxes_t3'] = bboxes_t3[j]
                    # self.labels[i]['bboxes_t4'] = bboxes_t4[j]
                    # self.labels[i]['bboxes_t5'] = bboxes_t5[j]
                    # self.labels[i]['bboxes_t6'] = bboxes_t6[j]
                else:
                    bboxes = self.labels[i]['bboxes']
                    segments = self.labels[i]['segments']
                    keypoints = self.labels[i]['keypoints']
                    j = (cls == include_class_array).any(1)
                    self.labels[i]['cls'] = cls[j]
                    self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        # i=1499
        im, f, fn = self.ims[i], self.im_files[i],self.npy_files[i]
        if self.cube:
            # todo
            fs = [self.im_files[i + x-self.frame_num +1] for x in range(self.frame_num)]
            vd_id = []
            for f in fs:
                vd_id.append(f.rsplit('/',1)[1].split('_')[-2])

            vd_change_id = self.frame_num - next((ii for ii, x in enumerate(vd_id) if x != vd_id[0]),-1)
            if vd_change_id < self.frame_num:

                fs = [self.im_files[i + x - vd_change_id -self.frame_num +1] for x in range(self.frame_num)]
                # saved_i = i  - vd_change_id -self.frame_num +1
                #视频文件错位矫正
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                if self.cube:
                    imgs = [None]*self.frame_num
                    if self.cube:
                        for ii, t in enumerate(fs):
                            if self.cube:
                                imgs[ii] = cv2.imread(t, cv2.IMREAD_GRAYSCALE)

                            try:
                                if not imgs[ii].all:
                                    print(f"image load error:{t}")
                            except:
                                print('t',t)
                            imgs[ii] = np.expand_dims(imgs[ii], axis=2)
                    else:
                        for ii, t in enumerate(fs):
                            imgs[ii] = cv2.imread(t)
                            if imgs[ii] == None:
                                print(f"image load error:{t}")
                    rgbimgs = cv2.imread(fs[-1])
                    imgs.insert(0, rgbimgs)
                    im = np.concatenate((imgs), axis=2)
                    self.fs_im = fs
                else:
                    im = cv2.imread(f)  # BGR
                    if im is None:
                        raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA

                # im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                #                 interpolation=interp)
                # 找到这一行:
                # im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

                if im.shape[2] > 4:
                    # 针对多通道 (>4) 的手动 Resize 逻辑
                    channel_list = []
                    for ii in range(im.shape[2]):
                        # 逐通道 Resize
                        res_channel = cv2.resize(im[:, :, ii], (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)), interpolation=cv2.INTER_LINEAR)
                        channel_list.append(res_channel)
                    im = np.stack(channel_list, axis=2)
                else:
                    # 原有的 OpenCV 逻辑 (保持 1-4 通道的性能)
                    im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)), interpolation=cv2.INTER_LINEAR)
            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    # def load_image(self, i):
    #     """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    #     if self.cube:
    #         img_path = f.rsplit('/', 1)[0]
    #         id = f.rsplit('/', 1)[1].split('.jpg')[0]
    #         # t6 = int(id)
    #         # t1 = max((t6 - 10), 1)
    #         # t2 = max((t6 - 8), 1)
    #         # t3 = max((t6 - 6), 1)
    #         # t4 = max((t6 - 4), 1)
    #         # t5 = max((t6 - 2), 1)
    #         t3 = int(id)
    #         t1 = max((t3 - 10), 1)
    #         t2 = max((t3 - 5), 1)
    #
    #         t1 = str(t1).zfill(len(id))
    #         t2 = str(t2).zfill(len(id))
    #         # t3 = str(t3).zfill(len(id))
    #         # t4 = str(t4).zfill(len(id))
    #         # t5 = str(t5).zfill(len(id))
    #         t1_path = img_path + '/' + t1 + '.jpg'
    #         t2_path = img_path + '/' + t2 + '.jpg'
    #         # t3_path = img_path + '/' + t3 + '.jpg'
    #         # t4_path = img_path + '/' + t4 + '.jpg'
    #         # t5_path = img_path + '/' + t5 + '.jpg'
    #         t3_path = f
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             im = np.load(fn)
    #         else:  # read image
    #             if self.cube:
    #                 if self.gray:
    #                     im3 = cv2.imread(t3_path, cv2.IMREAD_GRAYSCALE)
    #                 else:
    #                     im3 = cv2.imread(t3_path)
    #                 if im3 is None:
    #                     raise FileNotFoundError(f'Image Not Found {f}')
    #
    #                 # if not os.path.exists(t5_path):
    #                 #     im5 = im6
    #                 # else:
    #                 #     if self.gray:
    #                 #         im5 = cv2.imread(t5_path, cv2.IMREAD_GRAYSCALE)
    #                 #     else:
    #                 #         im5 = cv2.imread(t5_path)
    #
    #                 # if not os.path.exists(t4_path):
    #                 #     im4 = im5
    #                 # else:
    #                 #     if self.gray:
    #                 #         im4 = cv2.imread(t4_path, cv2.IMREAD_GRAYSCALE)
    #                 #     else:
    #                 #         im4 = cv2.imread(t4_path)
    #
    #                 # if not os.path.exists(t3_path):
    #                 #     im3 = im4
    #                 # else:
    #                 #     if self.gray:
    #                 #         im3 = cv2.imread(t3_path, cv2.IMREAD_GRAYSCALE)
    #                 #     else:
    #                 #         im3 = cv2.imread(t3_path)
    #
    #                 if not os.path.exists(t2_path):
    #                     im2 = im3
    #                 else:
    #                     if self.gray:
    #                         im2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
    #                     else:
    #                         im2 = cv2.imread(t2_path)
    #
    #                 if not os.path.exists(t1_path):
    #                     im1 = im2
    #                 else:
    #                     if self.gray:
    #                         im1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    #                     else:
    #                         im1 = cv2.imread(t1_path)
    #
    #                 if self.gray:
    #                     im1 = np.expand_dims(im1, axis=2)
    #                     im2 = np.expand_dims(im2, axis=2)
    #                     im3 = np.expand_dims(im3, axis=2)
    #                     # im4 = np.expand_dims(im4, axis=2)
    #                     # im5 = np.expand_dims(im5, axis=2)
    #                     # im6 = np.expand_dims(im6, axis=2)
    #                 im = np.concatenate((im1, im2, im3), axis=2)
    #             else:
    #                 im = cv2.imread(f)  # BGR
    #                 if im is None:
    #                     raise FileNotFoundError(f'Image Not Found {f}')
    #         h0, w0 = im.shape[:2]  # orig hw
    #         r = self.imgsz / max(h0, w0)  # ratio
    #         if r != 1:  # if sizes are not equal
    #             interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
    #             im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
    #                             interpolation=interp)
    #
    #         # Add to buffer if training with augmentations
    #         if self.augment:
    #             self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #             self.buffer.append(i)
    #             if len(self.buffer) >= self.max_buffer_length:
    #                 j = self.buffer.pop(0)
    #                 self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
    #
    #         return im, (h0, w0), im.shape[:2]
    #
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]
    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(results), total=self.ni, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        # index=3990
        # index = 3
        # print(index)
        # item1=self.get_image_and_label(index)
        # print(item1['im_file'])
        # item2=self.transforms(self.get_image_and_label(index))
        label=self.get_image_and_label(index)
        label=self.transforms(label)
        # save_visualized_batch(label)
        # visualize_and_save(label)
        return label

        # return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        # index=158
        # todo　视频切换，对齐标签

        if (index+self.frame_num)>len(self.im_files):
            index=len(self.im_files)-self.frame_num
            label = deepcopy(self.labels[-1])
        else:
            fs = [self.im_files[index + x -self.frame_num +1] for x in range(self.frame_num)]
            vd_id = []
            for f in fs:
                vd_id.append(f.rsplit('/', 1)[1].split('_')[-2])

            vd_change_id = self.frame_num - next((i for i, x in enumerate(vd_id) if x != vd_id[0]), -1)
            if vd_change_id < self.frame_num:
                # fs = [self.im_files[index + x - vd_change_id] for x in range(self.frame_num)]
                label = deepcopy(self.labels[index  - vd_change_id])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
                # 视频文件错位矫正
            else:
                label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label['checkfiles'].append(self.fs_im)
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        # print(label['checkfiles'])
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError
