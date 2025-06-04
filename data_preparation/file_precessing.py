import os
import shutil
import sys

def organize_files(source_directory):
    """
    将源目录中以'train'、'test'和'devel'开头的文件分别移动到对应的文件夹中。
    如果这些文件夹不存在，则创建它们。
    
    Args:
        source_directory: 包含文件的源目录路径
    """
    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误：目录 '{source_directory}' 不存在")
        return

    # 定义目标文件夹
    target_folders = {
        'train': os.path.join(source_directory, 'train_files'),
        'test': os.path.join(source_directory, 'test_files'),
        'devel': os.path.join(source_directory, 'devel_files')
    }
    
    # 创建目标文件夹（如果不存在）
    for folder in target_folders.values():
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"创建文件夹: {folder}")
    
    # 移动文件
    files_moved = {prefix: 0 for prefix in target_folders.keys()}
    
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        
        # 只处理文件（不处理目录）
        if os.path.isfile(file_path):
            for prefix, target_folder in target_folders.items():
                if filename.startswith(prefix):
                    target_path = os.path.join(target_folder, filename)
                    shutil.move(file_path, target_path)
                    files_moved[prefix] += 1
                    print(f"移动文件 '{filename}' 到 '{target_folder}'")
                    break
    
    # 打印摘要
    print("\n文件移动摘要:")
    for prefix, count in files_moved.items():
        print(f"{prefix} 文件: {count} 个文件已移动到 {target_folders[prefix]}")

if __name__ == "__main__":
    
    source_dir = "./ComParE2017_Cold_4students/wav"
    
    organize_files(source_dir)
    print("文件整理完成!")