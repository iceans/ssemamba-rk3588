import os
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请执行: pip install ultralytics opencv-python")

import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请检查 Ultralytics 库环境")


def test_via_validator(runs_base_dir: str, data_yaml_path: str, output_base_dir: str):
    """
    利用 Validator 管道运行 Test 集，规避 Predictor 的数据流异常
    """
    base_path = Path(runs_base_dir)
    out_path = Path(output_base_dir)

    # [条件判断] 确保 YAML 文件存在
    if not Path(data_yaml_path).exists():
        print(f"[Error] 未找到数据集配置文件: {data_yaml_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        weights_path = model_dir / "weights" / "best.pt"

        if not weights_path.exists():
            continue

        print(f"\n[Processing] 正在使用 Validation 管道评估: {model_name}")

        # try:
        model = YOLO(str(weights_path))

        # 关键改变：调用 val() 而非 predict()
        # 设置 split='test' 强制模型读取 dataset.yaml 中的 test 路径
        results = model.val(
            data=data_yaml_path,
            split='val',  # 指定评估测试集
            device="1",  # 硬件加速
            save_json=True,  # [可选] 保存所有预测框的具体坐标数据
            save_dir=str(out_path / model_name),
            plots=True,  # 生成类似 val_batch0_pred.jpg 的可视化图
            verbose=False,imgsz=1280,rect=False,frame_num=5
        )

        print(f"[Success] {model_name} 测试集验证完成，指标和批次图像已保存。")

        # except Exception as e:
        #     print(f"[Failed] {model_name} 验证管线运行失败: {e}")
        #     continue





if __name__ == "__main__":
    # ================= 路径配置区 =================
    # 根据实际情况修改以下路径

    # 包含 my_967, yolov5n_983, yolov8n_98 的上级目录
    RUNS_DIR = "/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/visualization/IRDST"

    # [需要补全] 测试图像所在目录
    TEST_IMAGES_DIR = "/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/IRDST_vis.yaml"

    # 预测结果保存目录d
    OUTPUT_DIR = "./visualization_results_IRDST"
    # ==============================================

    test_via_validator(RUNS_DIR, TEST_IMAGES_DIR, OUTPUT_DIR)