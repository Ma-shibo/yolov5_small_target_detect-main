#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估结果对比分析脚本
对比分析eval1到eval6的六次检测结果
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime
import matplotlib
import argparse

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def load_evaluation_data(base_dir):
    """加载所有评估结果数据"""
    base_path = Path(base_dir)
    evaluation_data = {}
    
    # 查找所有评估目录
    eval_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('eval')])
    
    print(f"找到 {len(eval_dirs)} 个评估目录:")
    for eval_dir in eval_dirs:
        print(f"  - {eval_dir.name}")
    
    for eval_dir in eval_dirs:
        eval_name = eval_dir.name
        config_file = eval_dir / 'config.yaml'
        results_file = eval_dir / 'evaluation_results.json'
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            eval_info = {
                'config': config_data,
                'has_results': results_file.exists()
            }
            
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    eval_info['results'] = json.load(f)
            
            evaluation_data[eval_name] = eval_info
        
    return evaluation_data

def extract_config_summary(evaluation_data):
    """提取配置信息摘要"""
    summary_data = []
    
    for eval_name, data in evaluation_data.items():
        config = data['config']
        
        # 提取预测目录的实验编号
        pred_dir = config['data_paths']['pred_dir']
        exp_match = pred_dir.split('/')[-2] if '/' in pred_dir else pred_dir
        
        summary = {
            'Evaluation': eval_name,
            'Experiment': exp_match,
            'Start Time': config['experiment_info']['start_time'],
            'GT Dir': config['data_paths']['gt_dir'].split('/')[-1],
            'Pred Dir': pred_dir.split('/')[-2:],
            'Has Results': data['has_results']
        }
        
        if data['has_results']:
            results = data['results']
            summary.update({
                'mAP@0.5': f"{results['mAP@0.5']:.4f}",
                'mAP@0.5:0.95': f"{results['mAP@0.5:0.95']:.4f}",
                'Precision': f"{results['Precision']:.4f}",
                'Recall': f"{results['Recall']:.4f}",
                'F1-Score': f"{results['F1-Score']:.4f}",
                'Total_TP': results['Total_TP'],
                'Total_FP': results['Total_FP'],
                'Total_FN': results['Total_FN']
            })
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)

def create_comparison_plots(evaluation_data, output_dir):
    """创建对比分析图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只处理有结果的评估
    valid_evals = {k: v for k, v in evaluation_data.items() if v['has_results']}
    
    if not valid_evals:
        print("⚠️  没有找到完整的评估结果文件")
        return
    
    eval_names = list(valid_evals.keys())
    
    # 提取指标数据
    metrics_data = {
        'mAP@0.5': [],
        'mAP@0.5:0.95': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    tp_data = []
    fp_data = []
    fn_data = []
    confidence_data = []
    
    for eval_name in eval_names:
        results = valid_evals[eval_name]['results']
        
        for metric in metrics_data:
            metrics_data[metric].append(results[metric])
        
        tp_data.append(results['Total_TP'])
        fp_data.append(results['Total_FP'])
        fn_data.append(results['Total_FN'])
        
        if 'confidence_stats' in results:
            conf_stats = results['confidence_stats']
            confidence_data.append({
                'eval': eval_name,
                'avg_all': conf_stats['avg_confidence'],
                'avg_tp': conf_stats['avg_tp_confidence'],
                'avg_fp': conf_stats['avg_fp_confidence']
            })
    
    # 1. 主要指标对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(eval_names))
    width = 0.15
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Evaluation Experiments', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detection Metrics Comparison Across Evaluations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(eval_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TP/FP/FN 对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(eval_names))
    width = 0.25
    
    ax.bar(x - width, tp_data, width, label='True Positive', color='#2ECC71', alpha=0.8)
    ax.bar(x, fp_data, width, label='False Positive', color='#E74C3C', alpha=0.8)
    ax.bar(x + width, fn_data, width, label='False Negative', color='#F39C12', alpha=0.8)
    
    ax.set_xlabel('Evaluation Experiments', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Detection Results Count Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(eval_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (tp, fp, fn) in enumerate(zip(tp_data, fp_data, fn_data)):
        ax.text(i - width, tp + 10, str(tp), ha='center', va='bottom', fontweight='bold')
        ax.text(i, fp + 10, str(fp), ha='center', va='bottom', fontweight='bold')
        ax.text(i + width, fn + 10, str(fn), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'counts_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 置信度对比图
    if confidence_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        conf_df = pd.DataFrame(confidence_data)
        x = np.arange(len(conf_df))
        width = 0.25
        
        ax.bar(x - width, conf_df['avg_all'], width, label='All Predictions', color='#3498DB', alpha=0.8)
        ax.bar(x, conf_df['avg_tp'], width, label='True Positive', color='#2ECC71', alpha=0.8)
        ax.bar(x + width, conf_df['avg_fp'], width, label='False Positive', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Evaluation Experiments', fontsize=12)
        ax.set_ylabel('Average Confidence', fontsize=12)
        ax.set_title('Confidence Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(conf_df['eval'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 趋势分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # mAP趋势
    ax1.plot(eval_names, metrics_data['mAP@0.5'], 'o-', linewidth=2, markersize=8, color='#FF6B6B', label='mAP@0.5')
    ax1.plot(eval_names, metrics_data['mAP@0.5:0.95'], 's-', linewidth=2, markersize=8, color='#4ECDC4', label='mAP@0.5:0.95')
    ax1.set_title('mAP Trend', fontweight='bold')
    ax1.set_ylabel('mAP Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Precision/Recall趋势
    ax2.plot(eval_names, metrics_data['Precision'], 'o-', linewidth=2, markersize=8, color='#45B7D1', label='Precision')
    ax2.plot(eval_names, metrics_data['Recall'], 's-', linewidth=2, markersize=8, color='#96CEB4', label='Recall')
    ax2.plot(eval_names, metrics_data['F1-Score'], '^-', linewidth=2, markersize=8, color='#FECA57', label='F1-Score')
    ax2.set_title('Precision/Recall/F1 Trend', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 检测数量趋势
    ax3.plot(eval_names, tp_data, 'o-', linewidth=2, markersize=8, color='#2ECC71', label='True Positive')
    ax3.plot(eval_names, fp_data, 's-', linewidth=2, markersize=8, color='#E74C3C', label='False Positive')
    ax3.plot(eval_names, fn_data, '^-', linewidth=2, markersize=8, color='#F39C12', label='False Negative')
    ax3.set_title('Detection Count Trend', fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 置信度趋势
    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        ax4.plot(conf_df['eval'], conf_df['avg_all'], 'o-', linewidth=2, markersize=8, color='#3498DB', label='All Predictions')
        ax4.plot(conf_df['eval'], conf_df['avg_tp'], 's-', linewidth=2, markersize=8, color='#2ECC71', label='True Positive')
        ax4.plot(conf_df['eval'], conf_df['avg_fp'], '^-', linewidth=2, markersize=8, color='#E74C3C', label='False Positive')
        ax4.set_title('Confidence Trend', fontweight='bold')
        ax4.set_ylabel('Average Confidence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trends_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比分析图表已保存到: {output_dir}")

def generate_comparison_report(evaluation_data, output_dir):
    """生成对比分析报告"""
    output_dir = Path(output_dir)
    
    # 配置信息摘要
    summary_df = extract_config_summary(evaluation_data)
    
    # 保存CSV文件
    summary_df.to_csv(output_dir / 'evaluation_summary.csv', index=False, encoding='utf-8')
    
    # 生成Markdown报告
    report_path = output_dir / 'comparison_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 检测结果评估对比分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 评估概览\n\n")
        f.write(f"- 总评估次数: {len(evaluation_data)}\n")
        f.write(f"- 完整结果数量: {sum(1 for v in evaluation_data.values() if v['has_results'])}\n\n")
        
        f.write("## 2. 配置信息摘要\n\n")
        f.write("| Evaluation | Experiment | Start Time | Has Results |\n")
        f.write("|------------|------------|------------|-------------|\n")
        
        for _, row in summary_df.iterrows():
            exp_dir = '/'.join(row['Pred Dir']) if isinstance(row['Pred Dir'], list) else row['Pred Dir']
            f.write(f"| {row['Evaluation']} | {exp_dir} | {row['Start Time']} | {'✅' if row['Has Results'] else '❌'} |\n")
        
        # 如果有完整结果，添加性能对比
        valid_evals = {k: v for k, v in evaluation_data.items() if v['has_results']}
        if valid_evals:
            f.write("\n## 3. 性能指标对比\n\n")
            f.write("| Evaluation | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | TP | FP | FN |\n")
            f.write("|------------|---------|--------------|-----------|--------|----------|----|----|----|\n")
            
            for eval_name, data in valid_evals.items():
                results = data['results']
                f.write(f"| {eval_name} | {results['mAP@0.5']:.4f} | {results['mAP@0.5:0.95']:.4f} | "
                       f"{results['Precision']:.4f} | {results['Recall']:.4f} | {results['F1-Score']:.4f} | "
                       f"{results['Total_TP']} | {results['Total_FP']} | {results['Total_FN']} |\n")
            
            # 找出最佳性能
            best_map = max(valid_evals.items(), key=lambda x: x[1]['results']['mAP@0.5'])
            best_f1 = max(valid_evals.items(), key=lambda x: x[1]['results']['F1-Score'])
            
            f.write(f"\n### 最佳性能\n\n")
            f.write(f"- **最高 mAP@0.5**: {best_map[0]} ({best_map[1]['results']['mAP@0.5']:.4f})\n")
            f.write(f"- **最高 F1-Score**: {best_f1[0]} ({best_f1[1]['results']['F1-Score']:.4f})\n")
            
            # 趋势分析
            f.write(f"\n### 趋势分析\n\n")
            
            eval_list = sorted(valid_evals.keys())
            if len(eval_list) >= 2:
                first_eval = valid_evals[eval_list[0]]['results']
                last_eval = valid_evals[eval_list[-1]]['results']
                
                map_change = last_eval['mAP@0.5'] - first_eval['mAP@0.5']
                f1_change = last_eval['F1-Score'] - first_eval['F1-Score']
                
                f.write(f"- mAP@0.5 变化: {first_eval['mAP@0.5']:.4f} → {last_eval['mAP@0.5']:.4f} "
                       f"({'↗️' if map_change > 0 else '↘️' if map_change < 0 else '➡️'} {map_change:+.4f})\n")
                f.write(f"- F1-Score 变化: {first_eval['F1-Score']:.4f} → {last_eval['F1-Score']:.4f} "
                       f"({'↗️' if f1_change > 0 else '↘️' if f1_change < 0 else '➡️'} {f1_change:+.4f})\n")
        
        f.write("\n## 4. 文件说明\n\n")
        f.write("- `evaluation_summary.csv`: 详细配置和结果数据\n")
        f.write("- `metrics_comparison.png`: 主要指标对比图\n")
        f.write("- `counts_comparison.png`: 检测数量对比图\n")
        f.write("- `confidence_comparison.png`: 置信度对比图\n")
        f.write("- `trends_analysis.png`: 趋势分析图\n")
    
    print(f"✅ 对比分析报告已保存: {report_path}")
    return summary_df

def print_summary(evaluation_data):
    """打印摘要信息"""
    print("\n" + "="*70)
    print("🔍 检测结果评估对比分析")
    print("="*70)
    
    print(f"📊 发现 {len(evaluation_data)} 个评估实验:")
    
    for eval_name, data in evaluation_data.items():
        config = data['config']
        start_time = config['experiment_info']['start_time']
        pred_dir = config['data_paths']['pred_dir']
        exp_name = pred_dir.split('/')[-2] if '/' in pred_dir else pred_dir
        
        status = "✅ 完整" if data['has_results'] else "⏳ 配置仅"
        print(f"  {eval_name}: {exp_name} ({start_time}) - {status}")
        
        if data['has_results']:
            results = data['results']
            print(f"    └─ mAP@0.5: {results['mAP@0.5']:.4f}, F1: {results['F1-Score']:.4f}, "
                 f"TP: {results['Total_TP']}, FP: {results['Total_FP']}, FN: {results['Total_FN']}")
    
    valid_count = sum(1 for v in evaluation_data.values() if v['has_results'])
    if valid_count > 0:
        print(f"\n📈 {valid_count}/{len(evaluation_data)} 个实验有完整结果可供对比分析")
    else:
        print(f"\n⚠️  没有找到完整的评估结果，建议先运行评估脚本生成结果")

def main():
    parser = argparse.ArgumentParser(description='对比分析多个评估结果')
    parser.add_argument('--eval-dir', type=str, 
                       default='runs/evaluation',
                       help='评估结果根目录')
    parser.add_argument('--output-dir', type=str,
                       default='runs/evaluation/comparison',
                       help='对比分析结果输出目录')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径
    base_dir = Path(args.eval_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not base_dir.exists():
        print(f"❌ 评估结果目录不存在: {base_dir}")
        return
    
    # 加载数据
    evaluation_data = load_evaluation_data(base_dir)
    
    if not evaluation_data:
        print(f"❌ 没有找到任何评估数据: {base_dir}")
        return
    
    # 打印摘要
    print_summary(evaluation_data)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成对比分析报告
    summary_df = generate_comparison_report(evaluation_data, output_dir)
    
    # 创建可视化图表
    create_comparison_plots(evaluation_data, output_dir)
    
    print(f"\n📁 完整对比分析结果已保存到: {output_dir}")
    print("包含以下文件:")
    print("  ├── comparison_report.md        # 对比分析报告")
    print("  ├── evaluation_summary.csv      # 详细数据表")
    print("  ├── metrics_comparison.png      # 指标对比图")
    print("  ├── counts_comparison.png       # 检测数量对比图")
    print("  ├── confidence_comparison.png   # 置信度对比图")
    print("  └── trends_analysis.png         # 趋势分析图")

if __name__ == '__main__':
    main()