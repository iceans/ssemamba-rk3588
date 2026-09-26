from pathlib import Path


def batch_remove_filename_suffix(
        target_dir: str,
        suffix_to_remove: str = "_pr_data"
):
    """
    遍历目标目录，安全移除特定名称后缀并保留扩展名。
    """
    target_path = Path(target_dir)

    if not target_path.exists() or not target_path.is_dir():
        print(f"[-] 目录不存在: {target_dir}")
        return

    # 1. 检索符合特征的文件
    # 获取所有包含特定后缀的 .npy 文件
    npy_files = list(target_path.glob(f"*{suffix_to_remove}.npy"))

    if not npy_files:
        print(f"[-] 未找到包含 '{suffix_to_remove}' 的 .npy 文件。")
        return

    print(f"[*] 发现 {len(npy_files)} 个待重命名文件, 开始执行...")

    # 2. 遍历执行安全替换
    for file_path in npy_files:
        original_name = file_path.name

        # 字符串精准替换: 仅移除目标后缀，确保 .npy 扩展名不受损
        new_name = original_name.replace(f"{suffix_to_remove}.npy", ".npy")
        new_name = new_name.split('_')[0]+'.npy'
        # 构建新的路径对象
        new_file_path = file_path.parent / new_name

        # 3. 触发重命名 IO 操作
        file_path.rename(new_file_path)
        print(f"    [重命名] {original_name} -> {new_name}")

    print("[*] 批量重命名完成。")


if __name__ == "__main__":
    # 执行配置: 指向存放 PR npy 数据的目录
    # 继承上一个脚本的数据持久化路径
    TARGET_DIR = "./"

    batch_remove_filename_suffix(
        target_dir=TARGET_DIR,
        suffix_to_remove="_pr_data"
    )