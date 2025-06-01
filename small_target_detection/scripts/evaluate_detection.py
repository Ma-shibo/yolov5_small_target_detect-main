#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测结果与人工标注对比分析脚本
计算检测准确性指标：Precision, Recall, F1-Score, mAP等
支持自动实验目录生成和丰富的可视化功能
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm
import argparse
import time
from datetime import datetime
import yaml
import glob
import re
import random

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def get_experiment_dir(project_dir, name='eval'):
    """获取实验目录，自动递增exp数字"""
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)
    
    # 查找现有的exp目录
    existing_dirs = list(project_path.glob(f'{name}*'))
    if not existing_dirs:
        exp_dir = project_path / f'{name}'
    else:
        # 提取数字并找到最大值
        numbers = []
        for dir_path in existing_dirs:
            if dir_path.name == name:
                numbers.append(0)
            else:
                match = re.search(rf'{name}(\d+)', dir_path.name)
                if match:
                    numbers.append(int(match.group(1)))
        
        next_num = max(numbers) + 1 if numbers else 1
        exp_dir = project_path / f'{name}{next_num}'
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def save_experiment_config(output_dir, args, start_time):
    """保存实验配置信息"""
    config = {
        'experiment_info': {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'script_path': str(Path(__file__).resolve()),
            'working_directory': str(Path.cwd()),
        },
        'data_paths': {
            'pred_dir': str(args.pred_dir),
            'gt_dir': str(args.gt_dir),
            'img_dir': str(args.img_dir),
            'output_dir': str(output_dir),
        },
        'parameters': {
            'iou_thresholds': args.iou_thresholds,
            'name': args.name,
            'project': args.project,
        }
    }
    
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def parse_yolo_label(label_path, img_width, img_height):
    """解析YOLO格式标签文件"""
    boxes = []
    if not Path(label_path).exists():
        return boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # 转换为xyxy格式
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            confidence = float(parts[5]) if len(parts) > 5 else 1.0
            
            boxes.append({
                'class_id': class_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': confidence
            })
    
    return boxes

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """匹配检测结果和真实标注"""
    matches = []
    unmatched_preds = list(range(len(pred_boxes)))
    unmatched_gts = list(range(len(gt_boxes)))
    
    # 按置信度排序预测框
    pred_indices = sorted(range(len(pred_boxes)), 
                         key=lambda i: pred_boxes[i]['confidence'], reverse=True)
    
    for pred_idx in pred_indices:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in unmatched_gts:
            if pred_boxes[pred_idx]['class_id'] == gt_boxes[gt_idx]['class_id']:
                iou = calculate_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'class_id': pred_boxes[pred_idx]['class_id']
            })
            unmatched_preds.remove(pred_idx)
            unmatched_gts.remove(best_gt_idx)
    
    return matches, unmatched_preds, unmatched_gts

def calculate_ap(precisions, recalls):
    """计算AP (Average Precision)"""
    # 添加端点
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # 计算precision的单调递减序列
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # 计算AP
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def evaluate_detection_results(pred_dir, gt_dir, img_dir, output_dir, 
                             iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                             conf_thresholds=np.arange(0.0, 1.01, 0.01)):
    """评估检测结果"""
    
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 开始评估检测结果...")
    print(f"📁 预测标签目录: {pred_dir}")
    print(f"📁 真实标签目录: {gt_dir}")
    print(f"📁 图片目录: {img_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(img_dir.glob(f'*{ext}')))
        image_files.extend(list(img_dir.glob(f'*{ext.upper()}')))
    
    print(f"📊 找到 {len(image_files)} 个图片文件")
    
    # 检查预测标签文件的存在情况
    matched_files = 0
    pred_files_found = []
    gt_files_found = []
    
    for img_file in image_files:
        img_name = img_file.stem
        pred_label = pred_dir / f"{img_name}.txt"
        gt_label = gt_dir / f"{img_name}.txt"
        
        if pred_label.exists():
            pred_files_found.append(img_name)
        if gt_label.exists():
            gt_files_found.append(img_name)
        if pred_label.exists() and gt_label.exists():
            matched_files += 1
    
    print(f"📊 找到预测标签文件: {len(pred_files_found)}")
    print(f"📊 找到真实标签文件: {len(gt_files_found)}")
    print(f"📊 匹配的文件对: {matched_files}")
    
    if matched_files == 0:
        print("⚠️  警告: 没有找到匹配的预测和真实标签文件!")
        print("请检查文件命名格式是否正确:")
        print(f"   预测标签格式: [图片名].txt")
        print(f"   真实标签格式: [图片名].txt")
        if pred_files_found:
            print(f"   找到的预测文件示例: {pred_files_found[:3]}")
        if gt_files_found:
            print(f"   找到的真实文件示例: {gt_files_found[:3]}")
    
    all_results = []
    class_stats = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0})
    confidence_stats = {'all_confidences': [], 'tp_confidences': [], 'fp_confidences': []}
    
    # 处理每个图片
    processed_images = 0
    for img_file in tqdm(image_files, desc="处理图片"):
        # 获取图片尺寸
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # 对应的标签文件 - 使用新的命名格式
        img_name = img_file.stem
        pred_label = pred_dir / f"{img_name}.txt"
        gt_label = gt_dir / f"{img_name}.txt"
        
        # 检查文件是否存在
        if not gt_label.exists():
            continue
            
        # 解析标签
        pred_boxes = parse_yolo_label(pred_label, img_width, img_height) if pred_label.exists() else []
        gt_boxes = parse_yolo_label(gt_label, img_width, img_height)
        
        processed_images += 1
        
        # 收集置信度统计
        for pred_box in pred_boxes:
            confidence_stats['all_confidences'].append(pred_box['confidence'])
        
        # 统计真实标注数量
        for gt_box in gt_boxes:
            class_stats[gt_box['class_id']]['num_gt'] += 1
        
        # 对不同IoU阈值进行匹配
        for iou_thresh in iou_thresholds:
            matches, unmatched_preds, unmatched_gts = match_detections(
                pred_boxes, gt_boxes, iou_thresh)
            
            # 记录TP和FP，以及对应的置信度
            for match in matches:
                pred_box = pred_boxes[match['pred_idx']]
                class_stats[match['class_id']]['tp'].append((pred_box['confidence'], iou_thresh))
                class_stats[match['class_id']]['scores'].append(pred_box['confidence'])
                if iou_thresh == 0.5:  # 只在IoU=0.5时记录置信度统计
                    confidence_stats['tp_confidences'].append(pred_box['confidence'])
            
            for pred_idx in unmatched_preds:
                pred_box = pred_boxes[pred_idx]
                class_stats[pred_box['class_id']]['fp'].append((pred_box['confidence'], iou_thresh))
                class_stats[pred_box['class_id']]['scores'].append(pred_box['confidence'])
                if iou_thresh == 0.5:  # 只在IoU=0.5时记录置信度统计
                    confidence_stats['fp_confidences'].append(pred_box['confidence'])
            
            # 记录图片级别的结果
            all_results.append({
                'image': img_name,
                'iou_threshold': iou_thresh,
                'tp': len(matches),
                'fp': len(unmatched_preds),
                'fn': len(unmatched_gts),
                'num_pred': len(pred_boxes),
                'num_gt': len(gt_boxes)
            })
    
    print(f"📊 实际处理了 {processed_images} 个图片文件")
    
    # 计算各类指标
    results = {}
    
    # 计算每个类别的AP
    class_aps = {}
    all_precisions = []
    all_recalls = []
    
    for class_id, stats in class_stats.items():
        if stats['num_gt'] == 0:
            continue
            
        class_aps[class_id] = {}
        
        for iou_thresh in iou_thresholds:
            # 获取该类别在该IoU阈值下的TP和FP
            tp_scores = [score for score, iou in stats['tp'] if iou == iou_thresh]
            fp_scores = [score for score, iou in stats['fp'] if iou == iou_thresh]
            
            # 合并并排序
            all_scores = [(score, 1) for score in tp_scores] + [(score, 0) for score in fp_scores]
            all_scores.sort(key=lambda x: x[0], reverse=True)
            
            if not all_scores:
                class_aps[class_id][iou_thresh] = 0.0
                continue
            
            # 计算precision和recall曲线
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            
            for score, is_tp in all_scores:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / stats['num_gt']
                
                precisions.append(precision)
                recalls.append(recall)
            
            # 计算AP
            ap = calculate_ap(np.array(precisions), np.array(recalls))
            class_aps[class_id][iou_thresh] = ap
            
            if iou_thresh == 0.5:  # 保存IoU=0.5的PR曲线数据
                all_precisions.extend(precisions)
                all_recalls.extend(recalls)
    
    # 计算mAP
    map_50 = np.mean([class_aps[cid][0.5] for cid in class_aps if 0.5 in class_aps[cid]])
    map_50_95 = np.mean([np.mean(list(class_aps[cid].values())) for cid in class_aps])
    
    # 计算整体指标
    total_tp = sum([len([1 for score, iou in stats['tp'] if iou == 0.5]) for stats in class_stats.values()])
    total_fp = sum([len([1 for score, iou in stats['fp'] if iou == 0.5]) for stats in class_stats.values()])
    total_gt = sum([stats['num_gt'] for stats in class_stats.values()])
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    results = {
        'mAP@0.5': map_50,
        'mAP@0.5:0.95': map_50_95,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1-Score': overall_f1,
        'Total_TP': total_tp,
        'Total_FP': total_fp,
        'Total_FN': total_gt - total_tp,
        'Total_GT': total_gt,
        'Total_Pred': total_tp + total_fp,
        'class_aps': class_aps
    }
    
    # 添加置信度分析到结果中
    results['confidence_stats'] = {
        'all_confidences': confidence_stats['all_confidences'],
        'tp_confidences': confidence_stats['tp_confidences'],
        'fp_confidences': confidence_stats['fp_confidences'],
        'avg_confidence': np.mean(confidence_stats['all_confidences']) if confidence_stats['all_confidences'] else 0,
        'avg_tp_confidence': np.mean(confidence_stats['tp_confidences']) if confidence_stats['tp_confidences'] else 0,
        'avg_fp_confidence': np.mean(confidence_stats['fp_confidences']) if confidence_stats['fp_confidences'] else 0
    }
    
    # 保存详细结果
    with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 创建可视化图表
    create_evaluation_plots(results, all_results, output_dir, confidence_stats)
    
    # 打印结果
    print_results(results)
    
    return results

def create_sample_images(pred_dir, gt_dir, img_dir, output_dir, max_samples=10):
    """创建样本图像可视化，显示预测vs真实标注的对比"""
    print("🖼️  生成样本图像可视化...")
    
    # 创建样本图像目录
    samples_dir = output_dir / 'sample_images'
    samples_dir.mkdir(exist_ok=True)
    
    # 获取有预测和真实标签的图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    matched_images = []
    
    for img_file in Path(img_dir).iterdir():
        if img_file.suffix.lower() in image_extensions:
            img_name = img_file.stem
            pred_label = Path(pred_dir) / f"{img_name}.txt"
            gt_label = Path(gt_dir) / f"{img_name}.txt"
            
            if pred_label.exists() and gt_label.exists():
                matched_images.append(img_file)
    
    # 随机选择样本
    random.seed(42)
    sample_images = random.sample(matched_images, min(max_samples, len(matched_images)))
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # BGR格式
    
    for i, img_file in enumerate(sample_images):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        img_name = img_file.stem
        
        # 读取预测和真实标签
        pred_label = Path(pred_dir) / f"{img_name}.txt"
        gt_label = Path(gt_dir) / f"{img_name}.txt"
        
        pred_boxes = parse_yolo_label(pred_label, img_width, img_height)
        gt_boxes = parse_yolo_label(gt_label, img_width, img_height)
        
        # 创建三个版本：原图、预测、真实、对比
        img_original = img.copy()
        img_pred = img.copy()
        img_gt = img.copy()
        img_comparison = img.copy()
        
        # 绘制预测框（红色）
        for box in pred_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img_comparison, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 添加置信度标签
            conf_text = f"{box['confidence']:.3f}"
            cv2.putText(img_pred, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img_comparison, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制真实框（绿色）
        for box in gt_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img_comparison, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 添加类别标签
            class_text = f"GT-{box['class_id']}"
            cv2.putText(img_gt, class_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img_comparison, class_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 创建组合图像
        combined = np.hstack([
            cv2.resize(img_original, (300, 300)),
            cv2.resize(img_pred, (300, 300)),
            cv2.resize(img_gt, (300, 300)),
            cv2.resize(img_comparison, (300, 300))
        ])
        
        # 添加标题
        title_height = 40
        title_img = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_img, "Original", (75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Predictions", (375, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Ground Truth", (675, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Comparison", (975, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        final_img = np.vstack([title_img, combined])
        
        # 保存图像
        output_path = samples_dir / f'sample_{i+1:02d}_{img_name}.jpg'
        cv2.imwrite(str(output_path), final_img)
    
    print(f"✅ 已生成 {len(sample_images)} 个样本图像到: {samples_dir}")
    return len(sample_images)

def create_error_analysis_images(pred_dir, gt_dir, img_dir, output_dir, all_results, max_samples=5):
    """创建错误分析图像，显示FP和FN的案例"""
    print("🔍 生成错误分析图像...")
    
    # 创建错误分析目录
    error_dir = output_dir / 'error_analysis'
    error_dir.mkdir(exist_ok=True)
    
    fp_cases = []  # 高FP的图像
    fn_cases = []  # 高FN的图像
    
    # 分析每张图片的FP和FN情况
    iou_05_results = [r for r in all_results if r['iou_threshold'] == 0.5]
    
    for result in iou_05_results:
        if result['fp'] > 0:
            fp_cases.append((result['image'], result['fp'], result['tp'], result['fn']))
        if result['fn'] > 0:
            fn_cases.append((result['image'], result['fn'], result['tp'], result['fp']))
    
    # 按错误数量排序
    fp_cases.sort(key=lambda x: x[1], reverse=True)
    fn_cases.sort(key=lambda x: x[1], reverse=True)
    
    # 生成FP案例图像
    for i, (img_name, fp_count, tp_count, fn_count) in enumerate(fp_cases[:max_samples]):
        img_path = Path(img_dir) / f"{img_name}.jpg"
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 创建对比图像：原图、预测、真实
        img_original = img.copy()
        img_pred = img.copy()
        img_gt = img.copy()
        
        # 读取对应标签
        pred_label = pred_dir / f"{img_name}.txt"
        gt_label = gt_dir / f"{img_name}.txt"
        
        pred_boxes = parse_yolo_label(pred_label, img.shape[1], img.shape[0]) if pred_label.exists() else []
        gt_boxes = parse_yolo_label(gt_label, img.shape[1], img.shape[0])
        
        # 绘制预测框（红色）
        for box in pred_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 绘制真实框（绿色）
        for box in gt_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 创建组合图像
        combined = np.hstack([
            cv2.resize(img_original, (300, 300)),
            cv2.resize(img_pred, (300, 300)),
            cv2.resize(img_gt, (300, 300))
        ])
        
        # 添加标题
        title_height = 40
        title_img = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_img, "Original", (75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Predictions", (375, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Ground Truth", (675, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        final_img = np.vstack([title_img, combined])
        
        # 保存图像
        output_path = error_dir / f'fp_case_{i+1:02d}_{img_name}.jpg'
        cv2.imwrite(str(output_path), final_img)
    
    # 生成FN案例图像
    for i, (img_name, fn_count, tp_count, fp_count) in enumerate(fn_cases[:max_samples]):
        img_path = Path(img_dir) / f"{img_name}.jpg"
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 创建对比图像：原图、预测、真实
        img_original = img.copy()
        img_pred = img.copy()
        img_gt = img.copy()
        
        # 读取对应标签
        pred_label = pred_dir / f"{img_name}.txt"
        gt_label = gt_dir / f"{img_name}.txt"
        
        pred_boxes = parse_yolo_label(pred_label, img.shape[1], img.shape[0]) if pred_label.exists() else []
        gt_boxes = parse_yolo_label(gt_label, img.shape[1], img.shape[0])
        
        # 绘制预测框（红色）
        for box in pred_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 绘制真实框（绿色）
        for box in gt_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 创建组合图像
        combined = np.hstack([
            cv2.resize(img_original, (300, 300)),
            cv2.resize(img_pred, (300, 300)),
            cv2.resize(img_gt, (300, 300))
        ])
        
        # 添加标题
        title_height = 40
        title_img = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_img, "Original", (75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Predictions", (375, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(title_img, "Ground Truth", (675, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        final_img = np.vstack([title_img, combined])
        
        # 保存图像
        output_path = error_dir / f'fn_case_{i+1:02d}_{img_name}.jpg'
        cv2.imwrite(str(output_path), final_img)
    
    print(f"✅ 已生成错误分析图像到: {error_dir}")

def create_evaluation_plots(results, all_results, output_dir, confidence_stats):
    """创建评估可视化图表"""
    print("📊 生成评估图表...")
    
    # 1. 整体指标条形图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    values = [results[metric] for metric in metrics]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Detection Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 检测结果分布饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # TP, FP, FN分布
    labels1 = ['True Positive', 'False Positive', 'False Negative']
    sizes1 = [results['Total_TP'], results['Total_FP'], results['Total_FN']]
    colors1 = ['#2ECC71', '#E74C3C', '#F39C12']
    
    ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Detection Results Distribution', fontsize=14, fontweight='bold')
    
    # 预测vs真实分布
    labels2 = ['Total Predictions', 'Total Ground Truth']
    sizes2 = [results['Total_Pred'], results['Total_GT']]
    colors2 = ['#3498DB', '#9B59B6']
    
    ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 各类别AP对比图
    if results['class_aps']:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        class_ids = list(results['class_aps'].keys())
        ap_50 = [results['class_aps'][cid].get(0.5, 0) for cid in class_ids]
        ap_avg = [np.mean(list(results['class_aps'][cid].values())) for cid in class_ids]
        
        x = np.arange(len(class_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ap_50, width, label='AP@0.5', color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, ap_avg, width, label='AP@0.5:0.95', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Class ID', fontsize=12)
        ax.set_ylabel('Average Precision', fontsize=12)
        ax.set_title('Average Precision by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {cid}' for cid in class_ids])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_ap_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. mAP vs IoU阈值曲线
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    map_values = []
    
    for iou_thresh in iou_thresholds:
        class_aps_at_iou = []
        for class_id in results['class_aps']:
            if iou_thresh in results['class_aps'][class_id]:
                class_aps_at_iou.append(results['class_aps'][class_id][iou_thresh])
        map_at_iou = np.mean(class_aps_at_iou) if class_aps_at_iou else 0
        map_values.append(map_at_iou)
    
    ax.plot(iou_thresholds, map_values, 'o-', linewidth=2, markersize=8, color='#3498DB')
    ax.set_xlabel('IoU Threshold', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('mAP vs IoU Threshold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 标注关键点
    for i, (iou, map_val) in enumerate(zip(iou_thresholds, map_values)):
        if iou in [0.5, 0.75, 0.9]:
            ax.annotate(f'{map_val:.3f}', (iou, map_val), textcoords="offset points",
                       xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'map_vs_iou.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 置信度分布直方图
    if confidence_stats['all_confidences']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 所有预测的置信度分布
        ax1.hist(confidence_stats['all_confidences'], bins=50, alpha=0.7, color='#3498DB', edgecolor='black')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('All Predictions Confidence Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        avg_conf = np.mean(confidence_stats['all_confidences'])
        ax1.axvline(avg_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_conf:.3f}')
        ax1.legend()
        
        # TP vs FP 置信度对比
        if confidence_stats['tp_confidences'] and confidence_stats['fp_confidences']:
            ax2.hist(confidence_stats['tp_confidences'], bins=30, alpha=0.7, color='#2ECC71', 
                    label=f'True Positive (n={len(confidence_stats["tp_confidences"])})', edgecolor='black')
            ax2.hist(confidence_stats['fp_confidences'], bins=30, alpha=0.7, color='#E74C3C', 
                    label=f'False Positive (n={len(confidence_stats["fp_confidences"])})', edgecolor='black')
            ax2.set_xlabel('Confidence Score', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('TP vs FP Confidence Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 添加平均值线
            if confidence_stats['tp_confidences']:
                tp_mean = np.mean(confidence_stats['tp_confidences'])
                ax2.axvline(tp_mean, color='#2ECC71', linestyle='--', linewidth=2, alpha=0.8)
            if confidence_stats['fp_confidences']:
                fp_mean = np.mean(confidence_stats['fp_confidences'])
                ax2.axvline(fp_mean, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. 置信度阈值 vs 精度/召回率曲线
    if confidence_stats['all_confidences']:
        # 收集不同置信度阈值下的性能数据
        conf_thresholds = np.arange(0.0, 1.01, 0.05)
        precisions = []
        recalls = []
        f1_scores = []
        
        for conf_thresh in conf_thresholds:
            tp_count = sum(1 for conf in confidence_stats['tp_confidences'] if conf >= conf_thresh)
            fp_count = sum(1 for conf in confidence_stats['fp_confidences'] if conf >= conf_thresh)
            fn_count = results['Total_FN']  # FN不受置信度阈值影响
            
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(conf_thresholds, precisions, 'o-', label='Precision', color='#3498DB', linewidth=2, markersize=4)
        ax.plot(conf_thresholds, recalls, 's-', label='Recall', color='#E74C3C', linewidth=2, markersize=4)
        ax.plot(conf_thresholds, f1_scores, '^-', label='F1-Score', color='#2ECC71', linewidth=2, markersize=4)
        
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # 找到最佳F1分数对应的置信度阈值
        best_f1_idx = np.argmax(f1_scores)
        best_conf_thresh = conf_thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        
        ax.axvline(best_conf_thresh, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Best F1 @ {best_conf_thresh:.2f}')
        ax.text(best_conf_thresh + 0.05, best_f1 - 0.05, f'F1={best_f1:.3f}', 
               fontsize=10, fontweight='bold', color='purple')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ 可视化图表已保存到: {output_dir}")
    print("新增置信度分析图表:")
    print("  ├── confidence_distribution.png     # 置信度分布图")
    print("  └── confidence_vs_performance.png   # 置信度阈值vs性能曲线")

def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("📊 检测结果评估报告")
    print("="*60)
    print(f"🎯 mAP@0.5:      {results['mAP@0.5']:.4f}")
    print(f"🎯 mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    print(f"📈 Precision:    {results['Precision']:.4f}")
    print(f"📈 Recall:       {results['Recall']:.4f}")
    print(f"📈 F1-Score:     {results['F1-Score']:.4f}")
    print("-"*60)
    print(f"✅ True Positive:  {results['Total_TP']}")
    print(f"❌ False Positive: {results['Total_FP']}")
    print(f"❌ False Negative: {results['Total_FN']}")
    print(f"📊 Total Ground Truth: {results['Total_GT']}")
    print(f"📊 Total Predictions:  {results['Total_Pred']}")
    
    # 添加置信度统计信息
    if 'confidence_stats' in results:
        conf_stats = results['confidence_stats']
        print("-"*60)
        print("🎯 置信度统计:")
        print(f"   平均置信度 (所有预测): {conf_stats['avg_confidence']:.4f}")
        print(f"   平均置信度 (True Positive): {conf_stats['avg_tp_confidence']:.4f}")
        print(f"   平均置信度 (False Positive): {conf_stats['avg_fp_confidence']:.4f}")
        print(f"   总预测数量: {len(conf_stats['all_confidences'])}")
        print(f"   TP数量: {len(conf_stats['tp_confidences'])}")
        print(f"   FP数量: {len(conf_stats['fp_confidences'])}")
        
        # 计算置信度分布统计
        if conf_stats['all_confidences']:
            confidences = np.array(conf_stats['all_confidences'])
            print(f"   置信度分布: 最小={confidences.min():.3f}, 最大={confidences.max():.3f}, 中位数={np.median(confidences):.3f}")
    
    print("="*60)
    
    if results['class_aps']:
        print("\n📋 各类别AP详情:")
        print("-"*40)
        for class_id, class_aps in results['class_aps'].items():
            ap_50 = class_aps.get(0.5, 0)
            ap_avg = np.mean(list(class_aps.values()))
            print(f"Class {class_id}: AP@0.5={ap_50:.4f}, AP@0.5:0.95={ap_avg:.4f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检测结果评估脚本')
    parser.add_argument('--pred-dir', type=str, 
                       default='/home/lkx/Documents/yolov5-v7/runs/detect/gui_test/exp14/labels',
                       help='预测标签目录')
    parser.add_argument('--gt-dir', type=str,
                       default='/home/lkx/Documents/1/2025_CUADC_Front_Camera_Concrete_Ground/train/labels',
                       help='真实标签目录')
    parser.add_argument('--img-dir', type=str,
                       default='/home/lkx/Documents/1/2025_CUADC_Front_Camera_Concrete_Ground/train/images',
                       help='图片目录')
    parser.add_argument('--output-dir', type=str,
                       default='runs/evaluation',
                       help='输出目录')
    parser.add_argument('--iou-thresholds', nargs='+', type=float,
                       default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                       help='IoU阈值列表')
    parser.add_argument('--name', type=str, default='eval', help='实验名称')
    parser.add_argument('--project', type=str, default='runs/evaluation', help='项目目录')
    
    args = parser.parse_args()
    
    # 创建实验目录
    output_dir = get_experiment_dir(Path(args.project), name=args.name)
    print(f"📂 创建实验目录: {output_dir}")
    
    # 保存实验配置
    start_time = datetime.now()
    save_experiment_config(output_dir, args, start_time)
    
    # 运行评估
    results = evaluate_detection_results(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        img_dir=args.img_dir,
        output_dir=output_dir,
        iou_thresholds=args.iou_thresholds
    )
    
    # 创建样本图像
    create_sample_images(args.pred_dir, args.gt_dir, args.img_dir, output_dir, max_samples=10)
    
    print(f"\n📁 详细结果已保存到: {output_dir}")
    print("包含以下文件:")
    print("  ├── evaluation_results.json    # 详细数值结果")
    print("  ├── overall_metrics.png        # 整体指标图")
    print("  ├── detection_distribution.png # 检测结果分布图")
    print("  ├── class_ap_comparison.png    # 各类别AP对比图")
    print("  └── map_vs_iou.png            # mAP vs IoU阈值曲线")
    print("  └── sample_images/            # 样本图像对比")
    print("  └── config.yaml               # 实验配置文件")

if __name__ == '__main__':
    main()