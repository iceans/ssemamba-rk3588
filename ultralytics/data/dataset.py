# Ultralytics YOLO 🚀, AGPL-3.0 license
# from itertools import repeat
from itertools import repeat, tee, islice
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

# from ..utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable
from ..utils import LOCAL_RANK, NUM_THREADS, is_dir_writeable
from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'
class YOLODataset(BaseDataset):
    """
    lxs修改数据命名格式规则：
    (*_视频名称_图片序号.*)图片／标签
    文件格式依照yolo要求
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        self.use_obb = False

        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)
    # origin version
    # def cache_labels(self, path=Path('./labels.cache')):
    #     """Cache dataset labels, check images and read shapes.
    #     Args:
    #         path (Path): path where to save the cache file (default: Path('./labels.cache')).
    #     Returns:
    #         (dict): labels.
    #     """
    #     x = {'labels': []}
    #     nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    #     desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
    #     total = len(self.im_files)
    #     nkpt, ndim = self.data.get('kpt_shape', (0, 0))
    #     if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
    #         raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
    #                          "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
    #     with ThreadPool(NUM_THREADS) as pool:
    #         results = pool.imap(func=verify_image_label,
    #                             iterable=zip(self.im_files, self.label_files, repeat(self.cube), repeat(self.prefix),
    #                                          repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
    #                                          repeat(ndim)))
    #
    #         ##todo
    #         # iters = tee(results, frame_num)
    #         pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
    #         for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
    #             nm += nm_f
    #             nf += nf_f
    #             ne += ne_f
    #             nc += nc_f
    #             if im_file:
    #                 if self.cube:
    #                     x['labels'].append(
    #                         dict(
    #                             im_file=im_file,
    #                             shape=shape,
    #                             cls=lb[:, 0:1],  # n, 1
    #                             bboxes_t1=lb[:, 1:5],  # n, 4
    #                             bboxes_t2=lb[:, 5:9],  # n, 4
    #                             bboxes_t3=lb[:, 9:13],  # n, 4
    #                             # bboxes_t4=lb[:, 13:17],
    #                             # bboxes_t5=lb[:, 17:21],
    #                             # bboxes_t6=lb[:, 21:25],
    #                             segments=segments,
    #                             keypoints=keypoint,
    #                             normalized=True,
    #                             bbox_format='xywh'))
    #                 else:
    #                     x['labels'].append(
    #                         dict(
    #                             im_file=im_file,
    #                             shape=shape,
    #                             cls=lb[:, 0:1],  # n, 1
    #                             bboxes=lb[:, 1:5],  # n, 4
    #                             segments=segments,
    #                             keypoints=keypoint,
    #                             normalized=True,
    #                             bbox_format='xywh'))
    #             if msg:
    #                 msgs.append(msg)
    #             pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
    #         pbar.close()
    #
    #     if msgs:
    #         LOGGER.info('\n'.join(msgs))
    #     if nf == 0:
    #         LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
    #     x['hash'] = get_hash(self.label_files + self.im_files)
    #     x['results'] = nf, nm, ne, nc, len(self.im_files)
    #     x['msgs'] = msgs  # warnings
    #     x['version'] = self.cache_version  # cache version
    #     if is_dir_writeable(path.parent):
    #         if path.exists():
    #             path.unlink()  # remove *.cache file if exists
    #         np.save(str(path), x)  # save cache for next time
    #         path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
    #         LOGGER.info(f'{self.prefix}New cache created: {path}')
    #     else:
    #         LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
    #     return x
    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        # self.frame_num = 3
        x = {'labels': []}

        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.cube), repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim), repeat(self.frame_num)))

            ##todo
            #self.frame_num need defined
            iters = list(tee(results, self.frame_num))
            for i,it in enumerate(iters):
                for n in range(i):
                    next(it)
                    # next(next(it))
            pbar = tqdm(zip(*iters), desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for window_tuple in pbar:
                im_file, lb_t1, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = window_tuple[self.frame_num-1]
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                im_files = []
                if im_file:
                    if self.cube:
                        label_data = dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb_t1[:, 0:1],  # n, 1
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh',
                            checkfiles=[]
                        )
                        for idx, item in enumerate(window_tuple):
                            key_name = f'bboxes_t{idx + 1}'
                            # if not item[1].any() :
                            #     label_data[key_name] = np.array([[0,0,0,0]])
                            # else:
                            #     lb_current = item[1]
                            #     label_data[key_name] = lb_current[:,1:5]

                            # if im_file=='/home/dell/lxs/JinJingBei/SatVideoIRSDT_yolo/train/images/IR_00009_06012.bmp':
                            #     pass
                            lb_current = item[1]
                            # if lb_current is None:
                            #     lb_current=[]
                            # if lb_current is not None and lb_current.shape[0] > 0:
                            # print(im_file)
                            label_data[key_name] = lb_current[:,1:5]
                            im_files.append(item[0])
                            # else:
                            #     print(lb_current)
                        label_data['checkfiles'].append(im_files)
                        x['labels'].append(label_data)

                    else:
                        x['labels'].append(
                            dict(
                                im_file=im_file,
                                shape=shape,
                                cls=lb[:, 0:1],  # n, 1
                                bboxes=lb[:, 1:5],  # n, 4
                                segments=segments,
                                keypoints=keypoint,
                                normalized=True,
                                bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()
        #todo 复制最后一组标签，补充为图像个数的相同长度
        if self.cube:
            if self.frame_num>1:
                for i in range(self.frame_num-1):
                    x['labels'].append(label_data)


        # from itertools import repeat
        # from multiprocessing.pool import ThreadPool
        # from tqdm import tqdm
        # import numpy as np
        # import os
        # # 1. 第一步：并行获取所有单帧的基础信息（不进行窗口滑动）
        # # 保持你原有的 verify_image_label 逻辑不变
        # with ThreadPool(NUM_THREADS) as pool:
        #     # 注意：这里 iterable 不需要 repeat frame_num，因为我们要先拿到纯粹的单帧列表
        #     # 假设 verify_image_label 返回: (im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg)
        #     results = list(tqdm(pool.imap(func=verify_image_label,
        #                                   iterable=zip(self.im_files, self.label_files,
        #                                                repeat(self.cube), repeat(self.prefix),
        #                                                repeat(self.use_keypoints), repeat(len(self.data['names'])),
        #                                                repeat(nkpt), repeat(ndim), repeat(1))),  # frame_num 传 1，先只校验单帧
        #                         desc=desc, total=total, bar_format=TQDM_BAR_FORMAT))
        #
        # # 2. 第二步：重组多帧逻辑 (Handle Video Boundaries & Alignment)
        # x['labels'] = []
        # msgs = []
        #
        # # 为了检测视频边界，我们需要判断文件夹变化
        # # 假设 im_file 的父目录代表视频 ID
        # get_video_id = lambda p: os.path.dirname(p) if p else None
        #
        # for i in range(len(results)):
        #     # 当前帧数据
        #     curr_res = results[i]
        #     im_file = curr_res[0]
        #
        #     # 统计信息累加 (nm, nf 等)
        #     nm += curr_res[5]
        #     nf += curr_res[6]
        #     ne += curr_res[7]
        #     nc += curr_res[8]
        #     if curr_res[9]: msgs.append(curr_res[9])
        #
        #     if im_file is None:
        #         continue  # 跳过损坏的图片
        #
        #     if self.cube:
        #         # --- 核心修复逻辑 ---
        #         # 我们以当前帧 i 为锚点，向前/向后寻找 frame_num 帧
        #         # 这里假设 logic 是：Current + History (即 t, t-1, t-2...)
        #         # 或者 Current + Future (即 t, t+1, t+2...)
        #         # TSG/Mamba 类模型通常需要时序连续性。
        #
        #         # 策略：构建窗口 [i - frame_num + 1, i] (Looking Back)
        #         # 或者 [i, i + frame_num - 1] (Looking Forward)
        #
        #         # 下面演示 【Looking Back (历史窗口)】，这是最符合实时推理的逻辑
        #         # 窗口范围: [i - frame_num + 1, ..., i]
        #
        #         window_indices = []
        #         curr_vid = get_video_id(im_file)
        #
        #         # 尝试填充窗口
        #         for offset in range(self.frame_num - 1, -1, -1):  # 例如 frame_num=3: 2, 1, 0
        #             idx = i - offset
        #
        #             # 边界检查 1: 索引不能小于 0
        #             if idx < 0:
        #                 valid_idx = i  # Padding 策略：如果越界，复制当前帧 (Replication Padding)
        #             else:
        #                 # 边界检查 2: 必须是同一个视频
        #                 prev_res = results[idx]
        #                 if prev_res[0] and get_video_id(prev_res[0]) == curr_vid:
        #                     valid_idx = idx
        #                 else:
        #                     # 如果跨越了视频边界（比如 idx 属于上一个视频），则复制当前视频的最早帧或当前帧
        #                     # 简单策略：复制当前帧 i
        #                     valid_idx = i
        #
        #             window_indices.append(valid_idx)
        #
        #         # 构建 label_data
        #         im_file, lb, shape, segments, keypoint, _, _, _, _, _ = results[i]
        #
        #         label_data = dict(
        #             im_file=im_file,
        #             shape=shape,
        #             cls=lb[:, 0:1],
        #             segments=segments,
        #             keypoints=keypoint,
        #             normalized=True,
        #             bbox_format='xywh'
        #         )
        #
        #         # 填入多帧 BBox
        #         for seq_idx, raw_idx in enumerate(window_indices):
        #             key_name = f'bboxes_t{seq_idx + 1}'  # bboxes_t1, t2, t3...
        #
        #             # 获取对应帧的 label
        #             target_res = results[raw_idx]
        #             lb_target = target_res[1]  # 假设 lb 在 index 1
        #
        #             if lb_target is not None and len(lb_target) > 0:
        #                 label_data[key_name] = lb_target[:, 1:5]
        #             else:
        #                 # 如果该帧没有目标 (空 label)，给一个空数组
        #                 label_data[key_name] = np.zeros((0, 4), dtype=np.float32)
        #
        #         x['labels'].append(label_data)
        #
        #     else:
        #         # 非 cube 模式 (Standard YOLO)
        #         im_file, lb, shape, segments, keypoint, _, _, _, _, _ = curr_res
        #         x['labels'].append(dict(
        #             im_file=im_file,
        #             shape=shape,
        #             cls=lb[:, 0:1],
        #             bboxes=lb[:, 1:5],
        #             segments=segments,
        #             keypoints=keypoint,
        #             normalized=True,
        #             bbox_format='xywh'
        #         ))
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_name = Path(str(Path(self.label_files[0]).parent)+f'_f{self.frame_num}')
        cache_path = cache_name.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        if self.cube:
            lengths = ((len(lb['cls']), len(lb['bboxes_t3']), len(lb['segments'])) for lb in labels)
        else:
            lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, cube=self.cube,frame_num=self.frame_num)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, cube=self.cube, frame_num=self.frame_num)])
            # transforms =Compose([])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask,
                   cube=self.cube,
                   frame_num=self.frame_num))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        if self.cube:
            for i in range(self.frame_num):
                name_instance =  f"instances_t{i+1}"
                name_box= f"bboxes_t{1+i}"
                bboxes = label.pop(name_box)
                label[name_instance] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format,normalized=normalized)
        else:
            bboxes = label.pop('bboxes')
            label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    # def update_labels_info(self, label):
    #     """custom your label format here."""
    #     # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
    #     # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
    #     if self.cube:
    #         bboxes_t1 = label.pop('bboxes_t1')
    #         bboxes_t2 = label.pop('bboxes_t2')
    #         bboxes_t3 = label.pop('bboxes_t3')
    #         # bboxes_t4 = label.pop('bboxes_t4')
    #         # bboxes_t5 = label.pop('bboxes_t5')
    #         # bboxes_t6 = label.pop('bboxes_t6')
    #     else:
    #         bboxes = label.pop('bboxes')
    #     segments = label.pop('segments')
    #     keypoints = label.pop('keypoints', None)
    #     bbox_format = label.pop('bbox_format')
    #     normalized = label.pop('normalized')
    #     if self.cube:
    #         label['instances_t1'] = Instances(bboxes_t1, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #         label['instances_t2'] = Instances(bboxes_t2, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #         label['instances_t3'] = Instances(bboxes_t3, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #         # label['instances_t4'] = Instances(bboxes_t4, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #         # label['instances_t5'] = Instances(bboxes_t5, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #         # label['instances_t6'] = Instances(bboxes_t6, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #     else:
    #         label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #     return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        frame_num=batch[0]['img'].shape[0]-3
        bboxes_name=[f'bboxes_t{num_n+1}' for num_n in range(frame_num)]
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                # value = torch.stack(value, 0)
                if isinstance(value[0], torch.Tensor):
                    # 1. 获取当前 Batch 中最大的高度和宽度
                    raw_max_h = max([t.shape[1] for t in value])
                    raw_max_w = max([t.shape[2] for t in value])

                    # 2. 计算目标尺寸：必须是 32 的倍数 (向上取整)
                    # 比如: 329 -> (329 + 31) // 32 * 32 = 352
                    stride = 32
                    target_h = (raw_max_h + stride - 1) // stride * stride
                    target_w = (raw_max_w + stride - 1) // stride * stride

                    padded_value = []
                    for t in value:
                        h, w = t.shape[1], t.shape[2]
                        # 计算需要补多少像素
                        pad_h = target_h - h
                        pad_w = target_w - w

                        if pad_h > 0 or pad_w > 0:
                            # value=0.447 是 YOLO 的灰色背景，也可以用 0
                            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0.447)
                        padded_value.append(t)
                    value = torch.stack(padded_value, 0)
                else:
                    value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']+ bboxes_name:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py
class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
class YOLOConcatDataset(ConcatDataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same collation
    function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)
# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
