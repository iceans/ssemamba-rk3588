import numpy as np
from pathlib import Path


def convert_to_dict_npy(input_file: str, output_file: str):
    """
    将 (101, 2) 的二维数组转换为包含 {'recall': [...], 'precision': [...]} 的字典格式 .npy
    """
    # 1. 加载原始二维矩阵数据
    # 你的 STEM.npy 是基础 Numpy 数组，此处无需 allow_pickle=True
    try:
        raw_matrix = np.load(input_file)
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {e}")

    # 2. 校验维度
    if len(raw_matrix.shape) != 2 or raw_matrix.shape[1] < 2:
        raise ValueError(f"输入数据维度异常，期望 (*, 2)，实际为 {raw_matrix.shape}")

    # 3. 提取特征列
    # 假设第 0 列为 Recall, 第 1 列为 Precision
    # 风险提示：若后续绘图发现曲线 x/y 轴物理意义倒置，请将此处的 0 和 1 互换
    recall_array = raw_matrix[:, 0]
    precision_array = raw_matrix[:, 1]

    # 4. 构建符合目标脚本规范的字典结构
    formatted_data = {
        'recall': recall_array,
        'precision': precision_array
    }

    # 5. 保存为包含字典的 0-d 数组
    # 必须开启 allow_pickle=True，否则 Numpy 无法序列化并保存 Python Dict 对象
    np.save(output_file, formatted_data, allow_pickle=True)
    print(f"[+] 转换完成: {input_file} -> {output_file}")

    # --- 防御性验证测试 ---
    test_load = np.load(output_file, allow_pickle=True).item()
    print(f"[+] 结构验证成功，当前 Keys: {list(test_load.keys())}")


if __name__ == "__main__":
    # 配置你的输入与输出路径
    input_path = "STEM.npy"
    output_path = "STEM.npy"

    # 如果需要批量转换，可以使用 pathlib.Path glob 配合 for 循环调用此函数
    convert_to_dict_npy(input_path, output_path)