import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# ================= 配置区域 (Configuration) =================
# [身份验证] 适配 Ubuntu 路径，请替换为你的实际图片路径
IMAGE_PATH = 'bus.jpg'  # 如果没有图片，脚本会自动下载一张示例图
MODEL_NAME = 'yolov8n.pt'  # 使用 nano 版本演示，方便快速加载，可换成 yolov8x.pt
OUTPUT_DIR = 'feature_maps_output'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ==========================================================

class FeatureMapExtractor:
    """
    [学术化封装] 用于通过 PyTorch Hook 机制提取 YOLOv8 指定层的特征图。
    Ref: PyTorch Hooks Mechanism for Intermediate Layer Extraction.
    """

    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hooks = []
        self.features = {}  # 存储提取的特征张量

    def _hook_fn(self, module, input, output, name):
        """
        Hook 回调函数：在 forward pass 期间自动被调用。
        :param output: 层的输出张量 (Batch, Channel, Height, Width)
        """
        # 仅保留输出张量到 CPU，避免显存爆炸 (Memory Leak Prevention)
        if isinstance(output, torch.Tensor):
            self.features[name] = output.detach().cpu()

    def register_hooks(self, target_layers=(nn.Conv2d,)):
        """
        递归注册 Hook 到所有指定类型的子模块。
        :param target_layers: 需要监控的层类型，默认为卷积层
        """
        print(f"[*] 正在注册 Hooks (目标层类型: {target_layers})...")
        for name, module in self.model.named_modules():
            # 过滤掉顶层容器，只关注具体算子
            if isinstance(module, target_layers):
                # 使用闭包保存 layer name
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self._hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)
        print(f"[*] 成功注册 {len(self.hooks)} 个 Hook。")

    def save_feature_maps(self):
        """
        [可视化逻辑] 将高维 Tensor 压缩为热力图并保存。
        逻辑：Channel-wise Average -> Normalize -> Apply Colormap -> Save
        """
        print(f"[*] 开始生成特征图可视化，保存路径: {self.output_dir}")
        count = 0
        for name, tensor in self.features.items():
            # tensor shape: [1, C, H, W] -> 压缩为 [H, W]
            # 方法：对通道维度求平均 (Average Pooling across channels)
            heatmap = torch.mean(tensor, dim=1).squeeze().numpy()

            # 归一化到 [0, 255] 以适配图像格式
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)

            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            # 应用热力图伪彩色 (COLORMAP_JET)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # 文件名处理：替换非法字符 (e.g., "model.0.conv" -> "model_0_conv")
            safe_name = name.replace('.', '_')
            save_path = self.output_dir / f"{safe_name}.png"

            cv2.imwrite(str(save_path), heatmap_color)
            count += 1

            # 仅打印前5个，避免刷屏
            if count <= 5:
                print(f"    -> 已保存: {save_path}")

        print(f"[*] 处理完成。共保存 {count} 层特征图。")

    def remove_hooks(self):
        """清理 Hooks，释放资源"""
        for h in self.hooks:
            h.remove()
        print("[*] Hooks 已清理。")


def main():
    # 0. 准备环境
    if not os.path.exists(IMAGE_PATH):
        # 下载示例图 (Using ultralytics utils)
        from ultralytics.utils.downloads import download
        download('https://ultralytics.com/images/bus.jpg')

    print(f"[*] 加载模型: {MODEL_NAME} 到 {DEVICE}")
    model = YOLO(MODEL_NAME)

    # -------------------------------------------------------------
    # 方案 A: 官方内置可视化 (High ROI, 快速预览)
    # -------------------------------------------------------------
    print("\n=== 方案 A: 执行官方 visualize=True ===")
    # 官方会将结果保存在 runs/detect/predict/ 目录下
    model.predict(IMAGE_PATH, visualize=True, device=DEVICE, save=False)
    print("[*] 官方可视化已完成，请检查 runs/detect/predict/ 目录。")

    # -------------------------------------------------------------
    # 方案 B: 深度定制 Hook 提取 (Academic, 逐层分析)
    # -------------------------------------------------------------
    print("\n=== 方案 B: 执行自定义 Hook 特征提取 ===")
    # 获取底层的 PyTorch nn.Module
    pytorch_model = model.model

    extractor = FeatureMapExtractor(pytorch_model, OUTPUT_DIR)

    # 注册 Hook (监控所有 Conv2d 层)
    extractor.register_hooks(target_layers=(nn.Conv2d,))

    # 预处理图片 (调用 YOLO 内部预处理流程以保证输入分布正确)
    # 这里的逻辑是直接使用 model 的 __call__ 触发 forward，但我们需要手动处理输入
    # 为了简化，我们再次调用 model.predict，Hook 会自动在后台捕获数据
    print("[*] 运行推理以触发 Hooks...")
    model.predict(IMAGE_PATH, device=DEVICE, verbose=False)

    # 保存结果
    extractor.save_feature_maps()

    # 清理
    extractor.remove_hooks()
    print("\n[Done] 任务完成。")


if __name__ == "__main__":
    main()