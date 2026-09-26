import numpy as np
from ultralytics import YOLO


def extract_and_save_pr_curve(
        model_path: str,
        data_yaml: str,
        save_path: str = "./pr/ours_s.npy",
        task_type: str = "detect"  # 区分目标检测(detect)或实例分割(segment)
):
    """
    加载 YOLO 模型, 在验证集上计算指标, 并提取/保存 PR 曲线底层数组。
    """
    # 1. 实例化模型
    model = YOLO(model_path)
    # model = RTDETR(model_path)
    # 2. 执行验证集评估 (自动触发内置的匹配与评价机制)
    # 强烈建议将 imgsz 和 batch 保持与你平时测试一致
    metrics = model.val(data=data_yaml, plots=False,imgsz=1280,rect=False,frame_num=5,)

    # 3. 定位指标对象
    if task_type == "segment":
        metric_obj = metrics.seg
    else:
        metric_obj = metrics.box

    # 4. 提取 PR 曲线数据
    # curves_results[0] 是 PR 曲线的数据: [Recall_array, Precision_matrix, "Recall", "Precision"]
    pr_data = metric_obj.curves_results[0]

    recall_array = pr_data[0]  # px: Recall 值，形状 (1000,)
    precision_matrix = pr_data[1]  # py: Precision 矩阵，形状 (num_classes, 1000)

    # [挑战预设与修正]: YOLO 输出的 Precision 是每个类别的独立曲线。
    # 论文中通常展示的是整个模型在数据集上的宏平均 (Macro-average) PR 曲线。
    # 必须在类别维度 (axis=0) 取均值，将其降维成 1D 数组。
    if len(precision_matrix.shape) > 1 and precision_matrix.shape[0] > 1:
        precision_array = np.mean(precision_matrix, axis=0)
    else:
        precision_array = precision_matrix.flatten()

    # 5. 数据持久化 (Data Persistence)
    # 采用 np.save 封装为字典，供之前的 matplotlib 多模型对比脚本直接读取
    np.save(save_path, {
        'recall': recall_array,
        'precision': precision_array
    })

    print(f"[*] 数据提取完成.")
    print(f"    Recall shape: {recall_array.shape}")
    print(f"    Precision shape: {precision_array.shape} (均值化后)")
    print(f"[*] PR 曲线底层数据已保存至: {save_path}")


if __name__ == "__main__":
    # 执行示例: 替换为实际的权重与数据集配置文件
    extract_and_save_pr_curve(
        # model_path="/home/dell/lxs/tsgmamba/ultralytics-main/runs/detect/train94/weights/best.pt",
        # model_path="/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/my_s_9316/weights/best.pt",
        # data_yaml="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/DAUB.yaml",
        # save_path="/home/dell/lxs/tsgmamba/ultralytics-main/pr/my_s_9316",
        # task_type="detect"
        # model_path="/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/IRDST/my_2timMix_c2spa_9531/weights/best.pt",
        model_path="/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/IRDST/my_1280_944/weights/best.pt",
        data_yaml="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/IRDST.yaml",
        save_path="/home/dell/lxs/tsgmamba/ultralytics-main/pr_IRDST/Ours_s.npy",
        task_type="detect"
    )