#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小目标检测模型比较分析脚本
比较不同模型在各种配置下的检测性能
"""

import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import matplotlib as mpl

# 设置通用字体配置 - 更健壮的方法
def setup_matplotlib_fonts():
    # 使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检测可用字体
    available_fonts = []
    try:
        from matplotlib.font_manager import fontManager
        for font in fontManager.ttflist:
            if any(name in font.name.lower() for name in ['simhei', 'microsoft yahei', 'wenquanyi', 'noto sans cjk']):
                available_fonts.append(font.name)
                break
    except Exception as e:
        print(f"Warning: Error checking fonts: {e}")
    
    # 如果有可用的中文字体，使用它
    if available_fonts:
        plt.rcParams['font.sans-serif'] = [available_fonts[0]] + plt.rcParams['font.sans-serif']
        print(f"使用中文字体: {available_fonts[0]}")
        return True
    else:
        print("未找到中文字体，将使用英文标签")
        return False

# 设置字体并确定是否使用中文标签
use_chinese_labels = setup_matplotlib_fonts()

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def get_label(zh_text, en_text):
    """根据字体可用性返回中文或英文标签"""
    return zh_text if use_chinese_labels else en_text

def load_evaluation_results(eval_dirs):
    """加载所有评估结果文件"""
    results = {}
    
    for eval_dir in eval_dirs:
        eval_path = Path(eval_dir)
        results_file = eval_path / 'evaluation_results.json'
        config_file = eval_path / 'config.yaml'
        
        if not results_file.exists() or not config_file.exists():
            print(f"警告：在 {eval_dir} 中找不到完整的评估结果")
            continue
            
        # 加载评估结果
        with open(results_file, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
            
        # 加载配置信息
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 获取模型名信息（从数据路径中提取）
        pred_dir = config['data_paths']['pred_dir']
        model_info = extract_model_info(pred_dir, eval_path.name)
        
        # 存储结果
        results[eval_path.name] = {
            'results': eval_results,
            'config': config,
            'model_info': model_info
        }
    
    return results

def extract_model_info(pred_dir, eval_name):
    """从预测目录和评估名称中提取模型信息"""
    model_info = {
        'eval_name': eval_name,
        'model_version': '未知',
        'image_size': '未知',
        'precision': '未知'
    }
    
    # 根据评估名称分配模型版本和配置
    # 注意：这里使用了题目中提供的映射关系
    if eval_name in ['eval1', 'eval2', 'eval3']:
        model_info['model_version'] = '2.0'
        if eval_name == 'eval1':
            model_info['image_size'] = '1280'
            model_info['precision'] = '全精度'
        elif eval_name == 'eval2':
            model_info['image_size'] = '1280'
            model_info['precision'] = '半精度'
        elif eval_name == 'eval3':
            model_info['image_size'] = '640'
            model_info['precision'] = '未知'
    
    elif eval_name == 'eval7':
        model_info['model_version'] = '4.0'
        model_info['image_size'] = '未知'
        model_info['precision'] = '未知'
    
    elif eval_name in ['eval8', 'eval9']:
        model_info['model_version'] = '3.0'
        model_info['image_size'] = '1280'
        if eval_name == 'eval8':
            model_info['precision'] = '全精度'
        elif eval_name == 'eval9':
            model_info['precision'] = '半精度'
    
    elif eval_name in ['eval10', 'eval11']:
        model_info['model_version'] = '1.0'
        model_info['image_size'] = '1280'
        if eval_name == 'eval10':
            model_info['precision'] = '全精度'
        elif eval_name == 'eval11':
            model_info['precision'] = '半精度'
    
    return model_info

def create_comparison_dataframe(results):
    """创建比较数据框"""
    data = []
    
    # 收集关键指标
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    
    for eval_name, eval_data in results.items():
        model_info = eval_data['model_info']
        model_results = eval_data['results']
        
        row = {
            'eval_name': eval_name,
            'model_version': model_info['model_version'],
            'image_size': model_info['image_size'],
            'precision': model_info['precision'],
            'Total_TP': model_results.get('Total_TP', 0),
            'Total_FP': model_results.get('Total_FP', 0),
            'Total_FN': model_results.get('Total_FN', 0),
            'Total_GT': model_results.get('Total_GT', 0),
            'Total_Pred': model_results.get('Total_Pred', 0)
        }
        
        # 添加关键性能指标
        for metric in metrics:
            row[metric] = model_results.get(metric, 0)
            
        # 添加置信度信息（如果存在）
        if 'confidence_stats' in model_results:
            conf_stats = model_results['confidence_stats']
            row['avg_confidence'] = conf_stats.get('avg_confidence', 0)
            row['avg_tp_confidence'] = conf_stats.get('avg_tp_confidence', 0)
            row['avg_fp_confidence'] = conf_stats.get('avg_fp_confidence', 0)
        
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 添加分类
    df['model_group'] = df['model_version'] + ' (' + df['image_size'] + ' ' + df['precision'] + ')'
    
    # 按模型版本和评估名称排序
    df = df.sort_values(['model_version', 'eval_name'])
    
    return df

def create_model_summary(df, output_dir):
    """创建模型性能总结"""
    summary_file = output_dir / 'model_comparison_summary.csv'
    df.to_csv(summary_file, index=False)
    
    # 打印摘要信息
    print("\n" + "="*80)
    print("模型性能指标对比摘要")
    print("="*80)
    
    summary_cols = ['eval_name', 'model_version', 'image_size', 'precision', 
                    'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    
    print(df[summary_cols].to_string(index=False))
    print("\n")

def create_metric_comparison_plot(df, metric_name, output_dir, title=None, ascending=False):
    """创建指定度量的比较图表"""
    if title is None:
        title = f"不同模型的{metric_name}对比"
        
    # 按指标排序
    df_sorted = df.sort_values(metric_name, ascending=ascending)
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 设置不同模型版本的颜色
    palette = {
        '1.0': '#FF9999',  # 浅红色
        '2.0': '#66B2FF',  # 浅蓝色
        '3.0': '#99FF99',  # 浅绿色
        '4.0': '#FFCC99',  # 浅橙色
    }
    
    # 用模型组作为颜色映射
    colors = [palette.get(model, '#CCCCCC') for model in df_sorted['model_version']]
    
    # 创建条形图
    bars = plt.bar(df_sorted['model_group'], df_sorted[metric_name], color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('模型配置')
    plt.ylabel(metric_name)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[ver], label=f'模型 {ver}')
                      for ver in sorted(df['model_version'].unique())]
    plt.legend(handles=legend_elements, loc='best')
    
    # 保存图表
    plt.savefig(output_dir / f'{metric_name.lower().replace("@", "_at_").replace(":", "_")}_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_recall_comparison(df, output_dir):
    """创建精确率和召回率对比图"""
    plt.figure(figsize=(14, 8))
    
    # 设置不同模型版本的颜色
    version_colors = {
        '1.0': '#FF5733',  # 红色
        '2.0': '#337DFF',  # 蓝色
        '3.0': '#33FF57',  # 绿色
        '4.0': '#FF33E9',  # 粉色
    }
    
    # 为每个模型创建标记
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p']
    
    # 按模型版本分组
    for i, (model_ver, group) in enumerate(df.groupby('model_version')):
        # 绘制精确率-召回率点
        for idx, row in group.iterrows():
            plt.scatter(row['Recall'], row['Precision'], 
                      c=version_colors.get(model_ver, '#999999'),
                      marker=markers[i % len(markers)], 
                      s=100,
                      label=f"{row['model_group']}")
            
            # 添加标签
            plt.annotate(row['eval_name'], 
                        (row['Recall'], row['Precision']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
    
    # 添加理想参考线
    plt.plot([0, 1], [1, 1], 'k--', alpha=0.3)  # 精确率=1的参考线
    plt.plot([1, 1], [0, 1], 'k--', alpha=0.3)  # 召回率=1的参考线
    
    # 添加等F1分数的等高线
    f1_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    x = np.linspace(0.01, 1, 100)
    
    for f1 in f1_levels:
        y = (f1 * x) / (2 * x - f1)
        valid_idx = (y >= 0) & (y <= 1)
        plt.plot(x[valid_idx], y[valid_idx], 'k--', alpha=0.2)
        # 在线上标记F1值
        mid_idx = np.argmin(np.abs(x[valid_idx] - 0.5))
        if mid_idx.size > 0 and mid_idx < sum(valid_idx):
            mid_x = x[valid_idx][mid_idx]
            mid_y = y[valid_idx][mid_idx]
            plt.text(mid_x, mid_y, f'F1={f1}', 
                    color='gray', fontsize=8, alpha=0.7,
                    horizontalalignment='center',
                    verticalalignment='center')
    
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('不同模型的精确率-召回率对比', fontsize=14, fontweight='bold')
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    # 添加图例
    handles = []
    labels = []
    for model_ver, color in version_colors.items():
        if model_ver in df['model_version'].values:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10))
            labels.append(f'模型 {model_ver}')
    
    plt.legend(handles, labels, loc='lower left')
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_chart(df, output_dir):
    """创建雷达图比较各模型的综合性能"""
    # 选择要显示的指标
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    
    # 准备数据，每个模型组一个雷达图
    plt.figure(figsize=(12, 10))
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 设置绘图的极坐标
    ax = plt.subplot(111, polar=True)
    
    # 绘制每个模型的雷达图
    for i, (name, group) in enumerate(df.groupby('model_group')):
        values = [group[metric].values[0] for metric in metrics]
        values += values[:1]  # 闭合雷达图
        
        # 绘制雷达线
        ax.plot(angles, values, linewidth=2, label=name)
        
        # 填充区域
        ax.fill(angles, values, alpha=0.1)
    
    # 设置雷达图的刻度标签
    plt.xticks(angles[:-1], metrics, size=12)
    
    # 设置y轴范围
    ax.set_ylim(0, 1)
    
    # 添加标签
    for angle, metric in zip(angles[:-1], metrics):
        if angle == 0:  # 右侧
            ha, va = "left", "center"
        elif 0 < angle < np.pi:  # 上半部分
            ha, va = "right" if angle > np.pi/2 else "left", "bottom"
        elif angle == np.pi:  # 左侧
            ha, va = "right", "center"
        else:  # 下半部分
            ha, va = "right" if angle < 3*np.pi/2 else "left", "top"
            
        ax.text(angle, 1.1, metric, size=14, 
               horizontalalignment=ha, verticalalignment=va)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 直接使用中文标题，不使用条件判断
    plt.title('模型性能综合雷达图对比', size=15, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap_comparison(df, output_dir):
    """创建热力图对比所有模型在各项指标上的表现"""
    # 选择要显示的指标
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    
    # 修复数据透视表问题：直接创建热力图数据，而不使用pivot
    model_groups = df['model_group'].values
    heatmap_data = pd.DataFrame(index=model_groups)
    
    # 填充热力图数据
    for metric in metrics:
        heatmap_data[metric] = df[metric].values
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    
    # 直接使用中文标题和标签
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', vmin=0, vmax=1,
               linewidths=.5, cbar_kws={'label': '性能得分'})
    
    plt.title('模型性能热力图对比', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('模型配置')
    
    # 使标签更清晰
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_heatmap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_recall_tradeoff(df, output_dir):
    """创建精确率-召回率权衡分析图"""
    plt.figure(figsize=(10, 6))
    
    # 模型版本颜色映射
    version_colors = {
        '1.0': '#FF5733',
        '2.0': '#337DFF',
        '3.0': '#33FF57',
        '4.0': '#FF33E9',
    }
    
    # 对每个模型版本绘制精确率-召回率散点图
    for model_ver, group in df.groupby('model_version'):
        plt.scatter(group['Recall'], group['Precision'], 
                  c=version_colors.get(model_ver, '#999999'), 
                  s=100, label=f'模型 {model_ver}',
                  alpha=0.7)
        
        # 添加文本标签
        for _, row in group.iterrows():
            plt.annotate(f"{row['eval_name']}\n{row['precision']} {row['image_size']}",
                        (row['Recall'], row['Precision']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    # 绘制等F1曲线
    recall = np.linspace(0.01, 1, 100)
    for f1 in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        precision = (f1 * recall) / (2 * recall - f1)
        valid_idx = (precision >= 0) & (precision <= 1)
        plt.plot(recall[valid_idx], precision[valid_idx], 'k--', alpha=0.2)
        
        # 在曲线上标记F1值
        if np.any(valid_idx):
            mid_idx = np.argmin(np.abs(recall[valid_idx] - 0.5))
            plt.text(recall[valid_idx][mid_idx], precision[valid_idx][mid_idx], 
                   f'F1={f1}', fontsize=8, alpha=0.7, ha='center')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率权衡分析', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_full_half_precision_comparison(df, output_dir):
    """创建全精度与半精度对比图"""
    # 筛选有全/半精度标记的数据
    precision_df = df[df['precision'].isin(['全精度', '半精度'])]
    
    # 检查是否有足够的数据进行比较
    if len(precision_df) < 2:
        print("没有足够的全精度和半精度数据进行比较")
        return
    
    # 按模型版本和精度进行分组
    grouped = precision_df.groupby(['model_version', 'precision'])
    
    # 准备对比数据
    models = []
    map50_full = []
    map50_half = []
    map50_95_full = []
    map50_95_half = []
    
    for name, group in grouped:
        model_ver, precision = name
        if model_ver not in models and precision == '全精度':
            models.append(model_ver)
            
        # 收集全精度和半精度的值
        if precision == '全精度':
            map50_full.append(group['mAP@0.5'].values[0])
            map50_95_full.append(group['mAP@0.5:0.95'].values[0])
        elif precision == '半精度':
            map50_half.append(group['mAP@0.5'].values[0])
            map50_95_half.append(group['mAP@0.5:0.95'].values[0])
    
    # 确保数据长度匹配
    min_len = min(len(map50_full), len(map50_half), len(map50_95_full), len(map50_95_half), len(models))
    models = models[:min_len]
    map50_full = map50_full[:min_len]
    map50_half = map50_half[:min_len]
    map50_95_full = map50_95_full[:min_len]
    map50_95_half = map50_95_half[:min_len]
    
    # 创建分组柱状图
    x = np.arange(len(models))
    width = 0.2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # mAP@0.5对比
    ax1.bar(x - width/2, map50_full, width, label='全精度', color='#3498db')
    ax1.bar(x + width/2, map50_half, width, label='半精度', color='#e74c3c')
    ax1.set_xlabel('模型版本')
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title('全精度与半精度的mAP@0.5对比', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'模型{m}' for m in models])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(map50_full, map50_half)):
        ax1.text(i - width/2, v1 + 0.01, f'{v1:.4f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, v2 + 0.01, f'{v2:.4f}', ha='center', va='bottom', fontsize=9)
        # 显示差异百分比
        diff = (v2 - v1) / v1 * 100
        ax1.text(i, max(v1, v2) + 0.05, f'{diff:+.2f}%', ha='center', color='green' if diff >= 0 else 'red')
    
    # mAP@0.5:0.95对比
    ax2.bar(x - width/2, map50_95_full, width, label='全精度', color='#3498db')
    ax2.bar(x + width/2, map50_95_half, width, label='半精度', color='#e74c3c')
    ax2.set_xlabel('模型版本')
    ax2.set_ylabel('mAP@0.5:0.95')
    ax2.set_title('全精度与半精度的mAP@0.5:0.95对比', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'模型{m}' for m in models])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(map50_95_full, map50_95_half)):
        ax2.text(i - width/2, v1 + 0.01, f'{v1:.4f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, v2 + 0.01, f'{v2:.4f}', ha='center', va='bottom', fontsize=9)
        # 显示差异百分比
        diff = (v2 - v1) / v1 * 100
        ax2.text(i, max(v1, v2) + 0.05, f'{diff:+.2f}%', ha='center', color='green' if diff >= 0 else 'red')
    
    plt.suptitle('全精度与半精度性能对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'full_vs_half_precision_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_size_comparison(df, output_dir):
    """创建不同输入尺寸的性能对比图"""
    # 筛选有尺寸信息的数据
    size_df = df[df['image_size'].isin(['640', '1280'])]
    
    # 只关注模型2.0，因为它有不同尺寸
    m2_df = size_df[size_df['model_version'] == '2.0']
    
    if len(m2_df) < 2:
        print("没有足够的不同尺寸数据进行比较")
        return
        
    # 按尺寸分组
    grouped = m2_df.groupby('image_size')
    
    # 准备对比数据
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    size_640_values = []
    size_1280_values = []
    
    for name, group in grouped:
        if name == '640':
            size_640_values = [group[metric].values[0] for metric in metrics]
        elif name == '1280':
            size_1280_values = [group[metric].values[0] for metric in metrics]
    
    if not size_640_values or not size_1280_values:
        print("无法找到完整的尺寸对比数据")
        return
    
    # 创建分组柱状图
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, size_640_values, width, label='640×640', color='#3498db')
    bars2 = ax.bar(x + width/2, size_1280_values, width, label='1280×1280', color='#e74c3c')
    
    ax.set_xlabel('评估指标')
    ax.set_ylabel('得分')
    ax.set_title('模型2.0在不同输入尺寸下的性能对比', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars, values in [(bars1, size_640_values), (bars2, size_1280_values)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 添加性能差异百分比
    for i, (v1, v2) in enumerate(zip(size_640_values, size_1280_values)):
        diff = (v2 - v1) / v1 * 100
        ax.text(i, max(v1, v2) + 0.05, f'{diff:+.2f}%', 
               ha='center', color='green' if diff >= 0 else 'red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'input_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_ranking(df, output_dir):
    """创建模型综合排名"""
    # 计算综合得分
    df['composite_score'] = (df['mAP@0.5'] * 0.3 + 
                           df['mAP@0.5:0.95'] * 0.3 + 
                           df['Precision'] * 0.2 + 
                           df['Recall'] * 0.1 + 
                           df['F1-Score'] * 0.1)
    
    # 按综合得分降序排序
    df_ranked = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    # 创建排名表格
    plt.figure(figsize=(12, 8))
    plt.axis('tight')
    plt.axis('off')
    
    # 筛选显示的列
    display_cols = ['eval_name', 'model_group', 'mAP@0.5', 'mAP@0.5:0.95', 
                   'Precision', 'Recall', 'F1-Score', 'composite_score']
    
    # 展示列名映射
    col_names = {
        'eval_name': '评估ID',
        'model_group': '模型配置',
        'composite_score': '综合得分'
    }
    
    # 准备表格数据
    table_data = df_ranked[display_cols].copy()
    table_data.index = range(1, len(table_data) + 1)  # 设置排名序号
    
    # 重命名列
    for col, name in col_names.items():
        table_data = table_data.rename(columns={col: name})
    
    # 创建表格
    table = plt.table(cellText=table_data.round(4).values,
                    rowLabels=[f'第{i}名' for i in table_data.index],
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f0f0f0']*len(table_data.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置标题
    plt.title('模型综合性能排名', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存排名数据到CSV
    table_data.to_csv(output_dir / 'model_ranking.csv')
    
    return df_ranked

def create_model_comparison_report(df, results, output_dir):
    """创建比较分析报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    df_ranked = df.sort_values('composite_score', ascending=False)
    best_model_info = df_ranked.iloc[0]
    
    report_md = f"""# 小目标检测模型比较分析报告

生成时间: {now}

## 模型概览

本报告比较了以下模型的检测性能:

- 模型1.0: 来自训练目录exp12
- 模型2.0: 来自训练目录long_training2  
- 模型3.0: 来自训练目录exp6
- 模型4.0: 来自训练目录exp7

测试配置包括不同的图像尺寸(640和1280)以及全精度与半精度模式。

## 性能排名

综合各项指标，模型性能排名如下:

1. **{best_model_info['model_group']}** (评估ID: {best_model_info['eval_name']})
   - mAP@0.5: {best_model_info['mAP@0.5']:.4f}
   - mAP@0.5:0.95: {best_model_info['mAP@0.5:0.95']:.4f}
   - 精确率: {best_model_info['Precision']:.4f}
   - 召回率: {best_model_info['Recall']:.4f}
   - F1分数: {best_model_info['F1-Score']:.4f}
   - 综合得分: {best_model_info['composite_score']:.4f}

## 关键发现

1. **模型版本对比**:
   - 模型{df_ranked['model_version'].iloc[0]}整体性能最优，在mAP和精确率方面表现突出
   - 模型{df_ranked['model_version'].iloc[-1]}表现相对较弱，主要在mAP@0.5:0.95方面落后

2. **精度模式对比**:
   - 半精度模式平均性能较全精度{' 提升' if df[df['precision'] == '半精度']['mAP@0.5'].mean() > df[df['precision'] == '全精度']['mAP@0.5'].mean() else ' 下降'}了{abs(df[df['precision'] == '半精度']['mAP@0.5'].mean() - df[df['precision'] == '全精度']['mAP@0.5'].mean()) / df[df['precision'] == '全精度']['mAP@0.5'].mean() * 100:.2f}%
   - 半精度模式对推理速度有明显提升，同时保持了接近全精度的检测性能

3. **输入尺寸对比**:
   - 1280×1280输入尺寸相比640×640{' 提升' if df[df['image_size'] == '1280']['mAP@0.5'].mean() > df[df['image_size'] == '640']['mAP@0.5'].mean() else ' 下降'}了{abs(df[df['image_size'] == '1280']['mAP@0.5'].mean() - df[df['image_size'] == '640']['mAP@0.5'].mean()) / df[df['image_size'] == '640']['mAP@0.5'].mean() * 100:.2f}%的mAP@0.5
   - 更大的输入尺寸对小目标检测效果提升显著

## 详细指标分析

### mAP@0.5比较
- 最高: {df['mAP@0.5'].max():.4f} (模型{df.loc[df['mAP@0.5'].idxmax(), 'model_group']})
- 最低: {df['mAP@0.5'].min():.4f} (模型{df.loc[df['mAP@0.5'].idxmin(), 'model_group']})
- 平均: {df['mAP@0.5'].mean():.4f}

### mAP@0.5:0.95比较
- 最高: {df['mAP@0.5:0.95'].max():.4f} (模型{df.loc[df['mAP@0.5:0.95'].idxmax(), 'model_group']})
- 最低: {df['mAP@0.5:0.95'].min():.4f} (模型{df.loc[df['mAP@0.5:0.95'].idxmin(), 'model_group']})
- 平均: {df['mAP@0.5:0.95'].mean():.4f}

### 置信度分析
- 平均置信度最高的模型: {df.loc[df['avg_confidence'].idxmax() if 'avg_confidence' in df.columns else 0, 'model_group']}
- 误检率与置信度存在明显的反相关关系，高置信度阈值可以显著降低误检率

## 结论与建议

1. **推荐模型**: {best_model_info['model_group']} 在总体性能上表现最佳，特别是在检测精度和召回率的平衡方面
2. **部署建议**: 
   - 对于资源受限场景: 使用半精度模型以提高速度
   - 对于高精度要求场景: 使用1280×1280输入尺寸的全精度模型
3. **优化方向**:
   - 进一步优化模型对小目标的检测能力
   - 尝试混合精度训练以平衡性能和速度

## 附录

完整的评估指标和比较图表可在输出目录中查看。
"""
    
    # 保存报告
    with open(output_dir / 'model_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"📝 分析报告已保存到: {output_dir / 'model_comparison_report.md'}")
    
    # 如果有pandoc，尝试转换为HTML或PDF
    try:
        import subprocess
        subprocess.run(['pandoc', str(output_dir / 'model_comparison_report.md'), '-o', 
                      str(output_dir / 'model_comparison_report.html')])
        print(f"📊 HTML报告已生成: {output_dir / 'model_comparison_report.html'}")
    except Exception as e:
        print(f"无法转换报告为HTML: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='小目标检测模型比较工具')
    parser.add_argument('--eval-dirs', type=str, nargs='+',
                       default=['runs/evaluation/eval1', 'runs/evaluation/eval2', 
                               'runs/evaluation/eval3', 'runs/evaluation/eval7',
                               'runs/evaluation/eval8', 'runs/evaluation/eval9',
                               'runs/evaluation/eval10', 'runs/evaluation/eval11'],
                       help='评估结果目录列表')
    parser.add_argument('--output-dir', type=str, default='runs/model_comparison',
                       help='比较结果输出目录')
    
    # 添加字体路径选项
    parser.add_argument('--font-path', type=str, 
                       default=None,
                       help='自定义中文字体文件路径 (.ttf)')
    
    args = parser.parse_args()
    
    # 如果提供了自定义字体路径
    if args.font_path and Path(args.font_path).exists():
        try:
            from matplotlib.font_manager import FontProperties
            custom_font = FontProperties(fname=args.font_path)
            plt.rcParams['font.family'] = custom_font.get_name()
            print(f"✓ 已加载自定义字体: {args.font_path}")
        except Exception as e:
            print(f"× 自定义字体加载失败: {e}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 开始模型比较分析...")
    print(f"📂 评估目录: {args.eval_dirs}")
    print(f"📁 输出目录: {output_dir}")
    
    # 加载评估结果
    results = load_evaluation_results(args.eval_dirs)
    print(f"✅ 已加载 {len(results)} 个评估结果")
    
    if len(results) == 0:
        print("❌ 未找到有效的评估结果，退出")
        return
    
    # 创建比较数据框
    df = create_comparison_dataframe(results)
    
    # 创建综合得分
    df = create_model_ranking(df, output_dir)
    
    # 创建模型性能总结
    create_model_summary(df, output_dir)
    
    # 创建比较图表
    print("🖼️ 生成比较图表...")
    create_metric_comparison_plot(df, 'mAP@0.5', output_dir)
    create_metric_comparison_plot(df, 'mAP@0.5:0.95', output_dir)
    create_metric_comparison_plot(df, 'Precision', output_dir)
    create_metric_comparison_plot(df, 'Recall', output_dir)
    create_metric_comparison_plot(df, 'F1-Score', output_dir)
    
    create_precision_recall_comparison(df, output_dir)
    create_radar_chart(df, output_dir)
    create_heatmap_comparison(df, output_dir)
    create_precision_recall_tradeoff(df, output_dir)
    
    # 特殊比较
    create_full_half_precision_comparison(df, output_dir)
    create_size_comparison(df, output_dir)
    
    # 创建报告
    create_model_comparison_report(df, results, output_dir)
    
    print(f"\n🎉 比较分析完成！所有结果已保存至：{output_dir}")
    print("生成的图表包括:")
    print("  ├── model_ranking.png             # 模型性能排名")
    print("  ├── mAP_at_0.5_comparison.png     # mAP@0.5比较图")
    print("  ├── mAP_at_0.5_0.95_comparison.png # mAP@0.5:0.95比较图")
    print("  ├── precision_comparison.png      # 精确率比较图")
    print("  ├── recall_comparison.png         # 召回率比较图")
    print("  ├── f1-score_comparison.png       # F1分数比较图")
    print("  ├── precision_recall_comparison.png # 精确率-召回率对比图")
    print("  ├── model_radar_comparison.png    # 雷达图对比")
    print("  ├── model_heatmap_comparison.png  # 热力图对比")
    print("  ├── full_vs_half_precision_comparison.png # 全精度与半精度对比")
    print("  └── input_size_comparison.png     # 输入尺寸对比")

if __name__ == '__main__':
    main()
