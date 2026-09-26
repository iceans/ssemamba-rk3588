import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_batch_pr_curves_decoupled(data_dir: str, save_name: str = "batch_pr_curve.pdf"):
    """
    渲染与图例解耦方案: 先绘制所有曲线，最后拦截并重排图例 (Legend Handles)，
    并自动修正图例名称的大小写格式。
    """
    # 1. 预设的标准图例顺序与格式 (严格按照你的图片要求)
    desired_order = [
        "YOLOv5n", "YOLOv5s", "YOLOv8n", "YOLOv8s", "YOLO11n", "YOLO11s",
        "SCTransNet", "RTDETR-L", "MambaYolo", "PConv", "IRSTD-YOLO",
        "YOLO26n", "YOLO26s",  "Tridos", "SSTNet", "CoMoE", "STME","DFAR", "Ours"
    ]

    # 构建匹配字典: {小写名称: (排序权重, 标准显示名称)}
    # 用于后续对杂乱的文件名进行模糊查找和格式修正
    order_map = {name.lower(): (idx, name) for idx, name in enumerate(desired_order)}

    # 2. 路径初始化
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    npy_files = list(dir_path.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"在 {data_dir} 中未找到任何 .npy 文件。")

    # 3. 图表与样式初始化
    plt.figure(figsize=(8, 6))

    academic_colors = [
        'green', 'blue', 'black', '#F4A460', '#9932CC',
        '#8FBC8F', '#CD5C5C', '#8B3A62', '#00BFFF', '#D2691E'
    ]
    line_styles = ['-', '--', '-.', ':']
    color_idx = 0

    # 4. 无序遍历与渲染 (仅负责把线画上去)
    for file_path in npy_files:
        raw_model_name = file_path.stem  # 原始文件名，可能很杂乱

        try:
            data = np.load(file_path, allow_pickle=True).item()
            recall = data['recall']
            precision = data['precision']
        except Exception as e:
            print(f"[警告] 无法解析 {file_path.name}, 错误: {e}")
            continue

        # 样式路由: 只要文件名里带 ours 或 proposed 就加粗标红
        if 'ours' in raw_model_name.lower() or 'proposed' in raw_model_name.lower():
            plt.plot(recall, precision, label=raw_model_name, color='red',
                     linestyle='-', linewidth=2.0, zorder=10)
        else:
            c_idx = color_idx % len(academic_colors)
            s_idx = (color_idx // len(academic_colors)) % len(line_styles)
            plt.plot(recall, precision, label=raw_model_name, color=academic_colors[c_idx],
                     linestyle=line_styles[s_idx], linewidth=1.5, zorder=5)
            color_idx += 1

    # ==================== 核心优化部分 ====================
    # 5. 图例句柄拦截与重排
    handles, labels = plt.gca().get_legend_handles_labels()

    processed_items = []
    for handle, original_label in zip(handles, labels):
        lbl_lower = original_label.lower()
        matched = False

        # 遍历标准字典，进行子串模糊匹配 (例如 "yolov5s_epoch100" 能匹配到 "yolov5s")
        for tgt_name, (weight, exact_name) in order_map.items():
            if tgt_name in lbl_lower:
                # 匹配成功: 记录权重，并替换为标准的 exact_name
                processed_items.append((weight, exact_name, handle))
                matched = True
                break

        if not matched:
            # 如果完全匹配不到预设列表，放到最后，保留原文件名
            processed_items.append((float('inf'), original_label, handle))

    # 按权重进行升序排序
    processed_items.sort(key=lambda x: x[0])

    # 解包出排序且修正后的 handles 和 labels
    sorted_labels = [item[1] for item in processed_items]
    sorted_handles = [item[2] for item in processed_items]
    # ======================================================

    # 6. 学术格式规范化 (Formatting)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('PR Curve on IRDST', fontsize=16)

    # 注入重排后的图例数据
    plt.legend(sorted_handles, sorted_labels, loc="lower left", fontsize=9, ncol=2)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)

    # 7. 图像输出
    plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.savefig(save_name.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"[*] 绘图完成，共渲染 {len(npy_files)} 条曲线。")
    print(f"[*] 图例已按预设顺序与格式强制对齐。")

    plt.show()


if __name__ == "__main__":
    # 指定存放 .npy 文件的文件夹路径DFAR56.npy
    target_directory = "./pr_IRDST/re"
    savename = "IRDST.png"
    # savename = "DAUB.pdf"
    # target_directory = "./pr_DAUB"
    plot_batch_pr_curves_decoupled(data_dir=target_directory,save_name=savename)