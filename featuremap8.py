import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 修复 Linux 显示问题
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import shutil
import copy
from ultralytics.nn.modules import *
# ================= 配置区域 =================
ROOT='/home/dell/lxs/tsgmamba/ultralytics-main/runs/detect/train72'
# ROOT='/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/ablation/Ablation/DAUB/frame7'
# ROOT='/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/my_1fft_9415'
MODEL_PATH = ROOT+'/weights/best.pt'  # 基础模型
# IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/images/UAVSwarm-40_000159.jpg'  # 替换你的测试图片路径
# IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/ITSDT-15K/yolo_format/images/train/clip_002_00012.bmp'
# IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format/train/images/IR_00005_00010.bmp'
IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format/train/images/IR_00005_01678.bmp'
# IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format/test/images/IR_00018_00492.bmp'
# IMG_PATH = '/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format/test/images/IR_00011_00279.bmp'
root,id=IMG_PATH.rsplit('_',1)
id=int(id.split('.')[0])
ids=[id-6,id-5,id-4,id-3,id-2,id-1,id]
IMG_PATHS = []
for n,i in enumerate(ids):
    t = str(i).zfill(5)
    img_paths=root+'_'+t+'.bmp'
    # img_paths = root + '_' + t + '.bmp'
    IMG_PATHS.append(img_paths)
# IMG_PATHS = '/home/dell/lxs/Anti_UAV_dataset/UAVSwarm-dataset-master/yolo_dataset/test/images/UAVSwarm-40_000159.jpg'  # 替换你的测试图片路径

OUTPUT_DIR = Path(ROOT+'/feature3')
TARGET_LAYERS = (TimeMambaStem,C2f,CCCBlock,HGStem,HGBlock,AIFI,TMixSSM,SPPF,LearnableFrequencyModulator)  # 监控层类型
# TARGET_LAYERS = (TimeMambaStem,C2f,CCCBlock,Conv,HGStem,HGBlock,AIFI)
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 修复 Linux 显示问题




# ===========================================

def prepare_4_channel_input(paths, img_size=640):
    """
    读取4张图 -> 灰度 -> Resize -> 堆叠 -> Tensor
    返回: (1, 4, 640, 640) 的 Tensor
    """
    frames = []
    print(f"📂 正在处理输入帧...")
    for p in paths:
        # 强制读取为灰度图 (H, W)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"找不到图片: {p}")

        # Resize
        img = cv2.resize(img, (img_size, img_size))
        frames.append(img)
    frames.append(img)
    frames.append(img)
    frames.append(img)
    # 堆叠: List[(H,W)...] -> Numpy(4, H, W)
    stacked = np.stack(frames, axis=0)

    # 归一化 + 转 Tensor
    tensor = torch.from_numpy(stacked).float() / 255.0

    # 增加 Batch 维度: (1, 4, 640, 640)
    return tensor.unsqueeze(0)

def prepare_3_channel_input(paths, img_size=640):
    """
    读取4张图 -> 灰度 -> Resize -> 堆叠 -> Tensor
    返回: (1, 4, 640, 640) 的 Tensor
    """
    frames = []
    print(f"📂 正在处理输入帧...")
    # for p in paths:
        # 强制读取为灰度图 (H, W)
    img = cv2.imread(paths, cv2.IMREAD_COLOR_BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise FileNotFoundError(f"找不到图片: {paths}")

    # Resize
    img = cv2.resize(img, (img_size, img_size))
    img = img.transpose(2, 0, 1)  # (3, 640, 640)

    # 再转 Tensor 并增加 Batch 维度
    # img = torch.from_numpy(img).float().unsqueeze(0)

    # 堆叠: List[(H,W)...] -> Numpy(4, H, W)
    # stacked = np.stack(frames, axis=0)

    # 归一化 + 转 Tensor
    tensor = torch.from_numpy(img).float() / 255.0

    # 增加 Batch 维度: (1, 4, 640, 640)
    return tensor.unsqueeze(0)
def adapt_first_layer(model, new_in_channels=4):
    """
    [关键步骤] 修改模型第一层卷积，使其接受 4 通道输入
    原理：复制原权重的前3个通道均值来初始化第4个通道
    """
    # YOLOv8 的第一层通常是 model.model[0].conv
    first_layer = model.model[0].conv

    if first_layer.in_channels == new_in_channels:
        print("✅ 模型第一层已适配 4 通道，无需修改。")
        return

    print(f"🔧 正在修改第一层卷积: {first_layer.in_channels} -> {new_in_channels} channels")

    # 获取旧参数
    old_weight = first_layer.weight.data  # shape: (out, 3, k, k)
    old_bias = first_layer.bias

    # 创建新层
    new_layer = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=(old_bias is not None)
    )

    # 权重初始化策略：
    # 前3个通道复制原权重
    new_layer.weight.data[:, :3, :, :] = old_weight
    # 第4个通道使用前3个通道的均值 (保持初始分布一致)
    new_layer.weight.data[:, 3, :, :] = torch.mean(old_weight, dim=1)

    if old_bias is not None:
        new_layer.bias.data = old_bias.data

    # 替换模型中的层
    model.model[0].conv = new_layer
    print("✅ 第一层修改完成！")


def process_feature_map(feature_map):
    # (1, C, H, W) -> Heatmap
    fm = feature_map.squeeze(0).mean(dim=0).cpu().detach().numpy()
    fm = np.maximum(fm, 0)
    if fm.max() > 0:
        fm /= fm.max()
    fm = (fm * 255).astype(np.uint8)
    return cv2.applyColorMap(fm, cv2.COLORMAP_JET)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on {device}")

    # 1. 准备数据
    try:
        input_tensor = prepare_4_channel_input(IMG_PATHS).to(device)
        print(f"✅ 输入 Tensor 形状: {input_tensor.shape}")  # 应为 [1, 4, 640, 640]
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return

    # 2. 准备模型
    yolo = YOLO(MODEL_PATH)
    model = yolo.model.to(device)

    # [核心] 适配通道数
    # adapt_first_layer(model, new_in_channels=4)
    model.to(device)  # 修改层后需再次确认在 GPU

    # 3. 注册钩子
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            save_path = OUTPUT_DIR
            save_path.mkdir(exist_ok=True, parents=True)
            if len(output.shape) == 4:
                cv2.imwrite(str(save_path / f"{name}.png"), process_feature_map(output))

        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, TARGET_LAYERS):
            hooks.append(layer.register_forward_hook(get_activation(name)))

    # 4. 推理
    print("running inference...")
    with torch.no_grad():
        model(input_tensor)

    for h in hooks: h.remove()
    print(f"🎉 特征图已保存至: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()