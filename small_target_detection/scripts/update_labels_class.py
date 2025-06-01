#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修改YOLO标注文件类别ID脚本
将指定文件夹中的所有.txt标注文件的类别ID（第一列）统一改为0
"""

import os
import argparse
from pathlib import Path

def update_labels_in_directory(labels_dir, backup=True):
    """
    更新指定目录下所有.txt文件的类别ID为0
    
    Args:
        labels_dir (str): labels目录路径
        backup (bool): 是否创建备份
    
    Returns:
        tuple: (成功处理的文件数, 总文件数)
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"❌ 目录不存在: {labels_dir}")
        return 0, 0
    
    # 查找所有.txt文件
    txt_files = list(labels_path.glob("**/*.txt"))
    
    if not txt_files:
        print(f"⚠️  目录中没有找到.txt文件: {labels_dir}")
        return 0, 0
    
    print(f"📁 处理目录: {labels_dir}")
    print(f"📄 找到 {len(txt_files)} 个标注文件")
    
    success_count = 0
    
    for txt_file in txt_files:
        try:
            # 读取原始内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 如果文件为空，跳过
            if not lines:
                continue
            
            # 创建备份（如果需要）
            if backup:
                backup_file = txt_file.with_suffix('.txt.bak')
                if not backup_file.exists():
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
            
            # 处理每一行，将第一列（类别ID）改为0
            updated_lines = []
            modified = False
            
            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    updated_lines.append(line + '\n')
                    continue
                
                parts = line.split()
                if len(parts) >= 5:  # YOLO格式至少需要5列：class x y w h
                    original_class = parts[0]
                    if original_class != '0':
                        modified = True
                    parts[0] = '0'  # 将类别ID设为0
                    updated_lines.append(' '.join(parts) + '\n')
                else:
                    # 格式不正确的行保持原样
                    updated_lines.append(line + '\n')
            
            # 只有当文件被修改时才写入
            if modified:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理文件失败 {txt_file}: {e}")
    
    print(f"✅ 成功处理 {success_count}/{len(txt_files)} 个文件")
    return success_count, len(txt_files)

def main():
    parser = argparse.ArgumentParser(description='批量修改YOLO标注文件的类别ID为0')
    parser.add_argument('--dirs', nargs='+', required=True,
                       help='要处理的labels目录路径（可指定多个）')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件')
    
    args = parser.parse_args()
    
    total_success = 0
    total_files = 0
    
    print("🔄 开始批量修改labels文件类别ID...")
    print("="*60)
    
    for labels_dir in args.dirs:
        success, total = update_labels_in_directory(labels_dir, backup=not args.no_backup)
        total_success += success
        total_files += total
        print()
    
    print("="*60)
    print(f"📊 总计: 成功处理 {total_success}/{total_files} 个文件")
    
    if not args.no_backup:
        print("💾 原始文件已备份为 .txt.bak 格式")
    
    print("✨ 所有标注文件的类别ID已统一改为0")

if __name__ == '__main__':
    # 如果没有命令行参数，则使用默认的三个常见目录
    import sys
    if len(sys.argv) == 1:
        print("🔍 检测到可能的labels目录...")
        
        # 常见的labels目录
        possible_dirs = [
            '/home/lkx/Documents/yolov5-v7/runs/detect/gui_test/exp4/labels',
            '/home/lkx/Documents/yolov5-v7/runs/detect/gui_test/exp5/labels',
            '/home/lkx/Documents/yolov5-v7/runs/detect/gui_test/exp6/labels'
        ]
        
        existing_dirs = [d for d in possible_dirs if Path(d).exists()]
        
        if existing_dirs:
            print(f"找到以下labels目录:")
            for i, dir_path in enumerate(existing_dirs, 1):
                file_count = len(list(Path(dir_path).glob("**/*.txt")))
                print(f"  {i}. {dir_path} ({file_count} 个txt文件)")
            
            print("\n要处理这些目录吗？(y/n): ", end="")
            response = input().strip().lower()
            
            if response in ['y', 'yes', '是']:
                total_success = 0
                total_files = 0
                
                print("\n🔄 开始处理...")
                print("="*60)
                
                for labels_dir in existing_dirs:
                    success, total = update_labels_in_directory(labels_dir, backup=True)
                    total_success += success
                    total_files += total
                    print()
                
                print("="*60)
                print(f"📊 总计: 成功处理 {total_success}/{total_files} 个文件")
                print("💾 原始文件已备份为 .txt.bak 格式")
                print("✨ 所有标注文件的类别ID已统一改为0")
            else:
                print("❌ 操作已取消")
        else:
            print("❌ 没有找到常见的labels目录")
            print("请使用以下命令手动指定目录:")
            print("python update_labels_class.py --dirs /path/to/labels1 /path/to/labels2 /path/to/labels3")
    else:
        main()