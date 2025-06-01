#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控和可视化脚本
实时监控YOLOv5小目标检测训练过程
"""

import os
import sys
from pathlib import Path

# 添加YOLOv5路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 向上两级到yolov5-v7目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import time
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# 检测是否有显示环境，如果没有则使用非交互式后端
try:
    import matplotlib
    if os.environ.get('DISPLAY', '') == '':
        print("No display found. Using non-interactive Agg backend.")
        matplotlib.use('Agg')
    elif 'PYTEST_CURRENT_TEST' in os.environ:
        matplotlib.use('Agg')
    else:
        # 尝试使用默认后端，如果失败则回退到Agg
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                print("Interactive backend not available. Using Agg backend.")
                matplotlib.use('Agg')
except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# 设置中文字体 - 修复字体问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 禁用中文显示，使用英文标签
USE_CHINESE = False

class EnhancedTrainingMonitor:
    """增强版训练监控类，生成与exp9相同的详细图表"""
    
    def __init__(self, results_path, save_dir=None, update_interval=30, headless=False):
        """
        初始化增强监控器
        
        Args:
            results_path: results.csv文件路径
            save_dir: 图表保存目录
            update_interval: 更新间隔(秒)
            headless: 无头模式，只生成图片不显示
        """
        self.results_path = Path(results_path)
        self.save_dir = Path(save_dir) if save_dir else self.results_path.parent
        self.update_interval = update_interval
        self.headless = headless or (matplotlib.get_backend() == 'Agg')
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据
        self.data = None
        self.last_modified = 0
        
        # 设置图表样式
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        # 创建图表
        self.setup_plots()
        
        if self.headless:
            print("运行在无头模式，图表将只保存为文件")
    
    def setup_plots(self):
        """设置详细的图表布局"""
        # 创建主窗口
        self.fig = plt.figure(figsize=(20, 16))
        title = 'YOLOv5 Small Target Detection Training Monitor (Enhanced)' if not USE_CHINESE else 'YOLOv5 小目标检测训练监控 (增强版)'
        self.fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # 创建更复杂的子图布局
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 损失函数图 (左上角 - 大图)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        loss_title = 'Training & Validation Loss' if not USE_CHINESE else '训练和验证损失'
        self.ax_loss.set_title(loss_title, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # 2. mAP指标图 (右上角)
        self.ax_map = self.fig.add_subplot(gs[0, 2:])
        map_title = 'mAP Metrics Evolution' if not USE_CHINESE else 'mAP指标演进'
        self.ax_map.set_title(map_title, fontweight='bold')
        self.ax_map.set_xlabel('Epoch')
        self.ax_map.set_ylabel('mAP')
        self.ax_map.grid(True, alpha=0.3)
        
        # 3. 精确度和召回率图
        self.ax_pr = self.fig.add_subplot(gs[1, 0])
        pr_title = 'Precision & Recall' if not USE_CHINESE else '精确度和召回率'
        self.ax_pr.set_title(pr_title, fontweight='bold')
        self.ax_pr.set_xlabel('Epoch')
        self.ax_pr.set_ylabel('Value')
        self.ax_pr.grid(True, alpha=0.3)
        
        # 4. 学习率图
        self.ax_lr = self.fig.add_subplot(gs[1, 1])
        lr_title = 'Learning Rate Schedule' if not USE_CHINESE else '学习率调度'
        self.ax_lr.set_title(lr_title, fontweight='bold')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.grid(True, alpha=0.3)
        
        # 5. F1分数图
        self.ax_f1 = self.fig.add_subplot(gs[1, 2])
        f1_title = 'F1 Score Evolution' if not USE_CHINESE else 'F1分数演进'
        self.ax_f1.set_title(f1_title, fontweight='bold')
        self.ax_f1.set_xlabel('Epoch')
        self.ax_f1.set_ylabel('F1 Score')
        self.ax_f1.grid(True, alpha=0.3)
        
        # 6. 训练效率图
        self.ax_efficiency = self.fig.add_subplot(gs[1, 3])
        eff_title = 'Training Efficiency' if not USE_CHINESE else '训练效率'
        self.ax_efficiency.set_title(eff_title, fontweight='bold')
        self.ax_efficiency.set_xlabel('Epoch')
        self.ax_efficiency.set_ylabel('Loss Reduction Rate')
        self.ax_efficiency.grid(True, alpha=0.3)
        
        # 7. 损失分解图
        self.ax_loss_detail = self.fig.add_subplot(gs[2, :2])
        detail_title = 'Detailed Loss Components' if not USE_CHINESE else '详细损失组件'
        self.ax_loss_detail.set_title(detail_title, fontweight='bold')
        self.ax_loss_detail.set_xlabel('Epoch')
        self.ax_loss_detail.set_ylabel('Loss Value')
        self.ax_loss_detail.grid(True, alpha=0.3)
        
        # 8. 性能对比图
        self.ax_performance = self.fig.add_subplot(gs[2, 2:])
        perf_title = 'Performance Metrics Comparison' if not USE_CHINESE else '性能指标对比'
        self.ax_performance.set_title(perf_title, fontweight='bold')
        self.ax_performance.set_xlabel('Metrics')
        self.ax_performance.set_ylabel('Score')
        
        # 9. 统计信息面板
        self.ax_stats = self.fig.add_subplot(gs[3, :2])
        stats_title = 'Training Statistics & Analysis' if not USE_CHINESE else '训练统计分析'
        self.ax_stats.set_title(stats_title, fontweight='bold')
        self.ax_stats.axis('off')
        
        # 10. 趋势分析图
        self.ax_trend = self.fig.add_subplot(gs[3, 2:])
        trend_title = 'Training Trend Analysis' if not USE_CHINESE else '训练趋势分析'
        self.ax_trend.set_title(trend_title, fontweight='bold')
        self.ax_trend.set_xlabel('Training Progress (%)')
        self.ax_trend.set_ylabel('Normalized Metrics')
        self.ax_trend.grid(True, alpha=0.3)
        
    def load_data(self):
        """加载训练数据"""
        try:
            if not self.results_path.exists():
                return False
                
            # 检查文件是否更新
            current_modified = os.path.getmtime(self.results_path)
            if current_modified <= self.last_modified:
                return False
                
            self.last_modified = current_modified
            
            # 读取CSV文件
            self.data = pd.read_csv(self.results_path)
            self.data.columns = self.data.columns.str.strip()  # 清理列名
            
            # 数据清理和转换
            numeric_columns = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
                             'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 
                             'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 
                             'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2', 'lr/pg0']
            
            for col in numeric_columns:
                if col in self.data.columns:
                    # 转换为数值类型，无效值设为NaN
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # 去除全部为NaN的行（从某个epoch开始完全失败的情况）
            # 但保留至少有损失值的行
            valid_loss_cols = ['train/box_loss', 'train/obj_loss']
            valid_rows = False
            for col in valid_loss_cols:
                if col in self.data.columns:
                    valid_rows = valid_rows | self.data[col].notna()
            
            if valid_rows is not False:
                self.data = self.data[valid_rows]
            
            return True
            
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
    
    def update_plots(self):
        """更新所有图表"""
        if self.data is None or len(self.data) == 0:
            return
            
        try:
            # 清除所有子图
            for ax in [self.ax_loss, self.ax_lr, self.ax_map, self.ax_pr, 
                      self.ax_f1, self.ax_efficiency, self.ax_loss_detail, 
                      self.ax_performance, self.ax_stats, self.ax_trend]:
                ax.clear()
            
            epochs = range(len(self.data))
            
            # 1. 损失函数图
            loss_title = 'Training & Validation Loss' if not USE_CHINESE else '训练和验证损失'
            self.ax_loss.set_title(loss_title, fontweight='bold')
            
            loss_plotted = False
            if 'train/box_loss' in self.data.columns:
                self.ax_loss.plot(epochs, self.data['train/box_loss'], 'b-', label='Train Box Loss', linewidth=2)
                loss_plotted = True
            if 'train/obj_loss' in self.data.columns:
                self.ax_loss.plot(epochs, self.data['train/obj_loss'], 'r-', label='Train Obj Loss', linewidth=2)
                loss_plotted = True
            if 'train/cls_loss' in self.data.columns:
                self.ax_loss.plot(epochs, self.data['train/cls_loss'], 'g-', label='Train Cls Loss', linewidth=2)
                loss_plotted = True
            if 'val/box_loss' in self.data.columns:
                self.ax_loss.plot(epochs, self.data['val/box_loss'], 'c--', label='Val Box Loss', linewidth=2)
                loss_plotted = True
            if 'val/obj_loss' in self.data.columns:
                self.ax_loss.plot(epochs, self.data['val/obj_loss'], 'm--', label='Val Obj Loss', linewidth=2)
                loss_plotted = True
            if 'val/cls_loss' in self.data.columns:
                self.ax_loss.plot(epochs, self.data['val/cls_loss'], 'y--', label='Val Cls Loss', linewidth=2)
                loss_plotted = True
                
            self.ax_loss.set_xlabel('Epoch')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.grid(True, alpha=0.3)
            if loss_plotted:
                self.ax_loss.legend()
            
            # 2. 学习率图
            lr_title = 'Learning Rate Schedule' if not USE_CHINESE else '学习率调度'
            self.ax_lr.set_title(lr_title, fontweight='bold')
            if 'lr/pg0' in self.data.columns:
                self.ax_lr.plot(epochs, self.data['lr/pg0'], 'purple', linewidth=2)
            elif 'x/lr0' in self.data.columns:
                self.ax_lr.plot(epochs, self.data['x/lr0'], 'purple', linewidth=2)
            self.ax_lr.set_xlabel('Epoch')
            self.ax_lr.set_ylabel('Learning Rate')
            self.ax_lr.grid(True, alpha=0.3)
            self.ax_lr.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # 3. mAP指标图
            map_title = 'mAP Metrics Evolution' if not USE_CHINESE else 'mAP指标演进'
            self.ax_map.set_title(map_title, fontweight='bold')
            
            map_plotted = False
            if 'metrics/mAP_0.5' in self.data.columns:
                self.ax_map.plot(epochs, self.data['metrics/mAP_0.5'], 'orange', 
                               label='mAP@0.5', linewidth=2, marker='o', markersize=4)
                map_plotted = True
            if 'metrics/mAP_0.5:0.95' in self.data.columns:
                self.ax_map.plot(epochs, self.data['metrics/mAP_0.5:0.95'], 'red', 
                               label='mAP@0.5:0.95', linewidth=2, marker='s', markersize=4)
                map_plotted = True
                
            self.ax_map.set_xlabel('Epoch')
            self.ax_map.set_ylabel('mAP')
            self.ax_map.grid(True, alpha=0.3)
            if map_plotted:
                self.ax_map.legend()
            
            # 4. 精确度和召回率图
            pr_title = 'Precision & Recall' if not USE_CHINESE else '精确度 & 召回率'
            self.ax_pr.set_title(pr_title, fontweight='bold')
            
            pr_plotted = False
            if 'metrics/precision' in self.data.columns:
                self.ax_pr.plot(epochs, self.data['metrics/precision'], 'blue', 
                              label='Precision', linewidth=2, marker='^', markersize=4)
                pr_plotted = True
            if 'metrics/recall' in self.data.columns:
                self.ax_pr.plot(epochs, self.data['metrics/recall'], 'green', 
                              label='Recall', linewidth=2, marker='v', markersize=4)
                pr_plotted = True
                
            self.ax_pr.set_xlabel('Epoch')
            self.ax_pr.set_ylabel('Value')
            self.ax_pr.grid(True, alpha=0.3)
            if pr_plotted:
                self.ax_pr.legend()
            
            # 5. F1分数图
            f1_title = 'F1 Score Evolution' if not USE_CHINESE else 'F1分数演进'
            self.ax_f1.set_title(f1_title, fontweight='bold')
            
            if 'metrics/f1_score' in self.data.columns:
                self.ax_f1.plot(epochs, self.data['metrics/f1_score'], 'brown', 
                               label='F1 Score', linewidth=2, marker='x', markersize=4)
                self.ax_f1.set_ylim(0, 1)  # F1分数范围 [0, 1]
                self.ax_f1.legend()
            
            self.ax_f1.set_xlabel('Epoch')
            self.ax_f1.set_ylabel('F1 Score')
            self.ax_f1.grid(True, alpha=0.3)
            
            # 6. 训练效率图
            eff_title = 'Training Efficiency' if not USE_CHINESE else '训练效率'
            self.ax_efficiency.set_title(eff_title, fontweight='bold')
            
            if 'train/box_loss' in self.data.columns and 'val/box_loss' in self.data.columns:
                # 计算损失减少率
                self.data['box_loss_reduction'] = self.data['train/box_loss'] - self.data['val/box_loss']
                self.ax_efficiency.plot(epochs, self.data['box_loss_reduction'], 'green', 
                                       label='Loss Reduction', linewidth=2)
                self.ax_efficiency.set_ylabel('Loss Reduction')
                self.ax_efficiency.legend()
            
            self.ax_efficiency.set_xlabel('Epoch')
            self.ax_efficiency.grid(True, alpha=0.3)
            
            # 7. 损失分解图
            detail_title = 'Detailed Loss Components' if not USE_CHINESE else '详细损失组件'
            self.ax_loss_detail.set_title(detail_title, fontweight='bold')
            
            # 堆叠损失分解图
            if 'train/box_loss' in self.data.columns:
                self.ax_loss_detail.plot(epochs, self.data['train/box_loss'], 'b-', label='Train Box Loss', linewidth=2)
            if 'train/obj_loss' in self.data.columns:
                self.ax_loss_detail.plot(epochs, self.data['train/obj_loss'], 'r-', label='Train Obj Loss', linewidth=2)
            if 'train/cls_loss' in self.data.columns:
                self.ax_loss_detail.plot(epochs, self.data['train/cls_loss'], 'g-', label='Train Cls Loss', linewidth=2)
            if 'val/box_loss' in self.data.columns:
                self.ax_loss_detail.plot(epochs, self.data['val/box_loss'], 'c--', label='Val Box Loss', linewidth=2)
            if 'val/obj_loss' in self.data.columns:
                self.ax_loss_detail.plot(epochs, self.data['val/obj_loss'], 'm--', label='Val Obj Loss', linewidth=2)
            if 'val/cls_loss' in self.data.columns:
                self.ax_loss_detail.plot(epochs, self.data['val/cls_loss'], 'y--', label='Val Cls Loss', linewidth=2)
            
            self.ax_loss_detail.set_xlabel('Epoch')
            self.ax_loss_detail.set_ylabel('Loss Value')
            self.ax_loss_detail.grid(True, alpha=0.3)
            self.ax_loss_detail.legend()
            
            # 8. 性能对比图
            self.ax_performance.clear()
            self.ax_performance.axis('off')
            perf_title = 'Performance Metrics Comparison' if not USE_CHINESE else '性能指标对比'
            self.ax_performance.set_title(perf_title, fontweight='bold')
            
            if len(self.data) > 0:
                latest = self.data.iloc[-1]
                
                # 显示最新的mAP, Precision, Recall, F1 Score
                metrics = ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'metrics/precision', 'metrics/recall', 'metrics/f1_score']
                for i, metric in enumerate(metrics):
                    if metric in self.data.columns:
                        value = latest[metric] if pd.notna(latest[metric]) else 0
                        self.ax_performance.text(0.1, 0.9 - i*0.15, f"{metric.split('/')[-1]}: {value:.4f}",
                                             fontsize=12, verticalalignment='top', fontfamily='monospace',
                                             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # 9. 统计信息面板
            self.ax_stats.axis('off')
            stats_title = 'Training Statistics & Analysis' if not USE_CHINESE else '训练统计分析'
            self.ax_stats.set_title(stats_title, fontweight='bold')
            
            current_epoch = len(self.data)
            stats_text = f"Current Epoch: {current_epoch}\n"
            stats_text += f"Update Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            if len(self.data) > 0:
                latest = self.data.iloc[-1]
                
                # 最新损失值
                if 'train/box_loss' in self.data.columns:
                    stats_text += f"Box Loss: {latest['train/box_loss']:.4f}\n"
                if 'train/obj_loss' in self.data.columns:
                    stats_text += f"Obj Loss: {latest['train/obj_loss']:.4f}\n"
                if 'train/cls_loss' in self.data.columns:
                    stats_text += f"Cls Loss: {latest['train/cls_loss']:.4f}\n"
                
                stats_text += "\n"
                
                # 最新指标
                if 'metrics/mAP_0.5' in self.data.columns:
                    stats_text += f"mAP@0.5: {latest['metrics/mAP_0.5']:.4f}\n"
                if 'metrics/mAP_0.5:0.95' in self.data.columns:
                    stats_text += f"mAP@0.5:0.95: {latest['metrics/mAP_0.5:0.95']:.4f}\n"
                if 'metrics/precision' in self.data.columns:
                    stats_text += f"Precision: {latest['metrics/precision']:.4f}\n"
                if 'metrics/recall' in self.data.columns:
                    stats_text += f"Recall: {latest['metrics/recall']:.4f}\n"
                if 'metrics/f1_score' in self.data.columns:
                    stats_text += f"F1 Score: {latest['metrics/f1_score']:.4f}\n"
                
                # 最佳值
                if 'metrics/mAP_0.5' in self.data.columns:
                    best_map = self.data['metrics/mAP_0.5'].max()
                    best_epoch = self.data['metrics/mAP_0.5'].idxmax()
                    stats_text += f"\nBest mAP@0.5: {best_map:.4f}\n"
                    stats_text += f"Best Epoch: {best_epoch + 1}\n"
            
            self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                             fontsize=10, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # 10. 趋势分析图
            trend_title = 'Training Trend Analysis' if not USE_CHINESE else '训练趋势分析'
            self.ax_trend.set_title(trend_title, fontweight='bold')
            
            if len(self.data) > 0:
                # 计算训练进度百分比
                total_epochs = len(self.data)
                self.data['train_progress'] = (self.data.index + 1) / total_epochs * 100
                
                # 标准化指标用于趋势分析
                metrics_to_normalize = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 
                                       'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 
                                       'metrics/precision', 'metrics/recall', 'metrics/f1_score']
                
                normalized_data = self.data.copy()
                for metric in metrics_to_normalize:
                    if metric in self.data.columns:
                        # Min-Max标准化
                        min_val = self.data[metric].min()
                        max_val = self.data[metric].max()
                        normalized_data[metric] = (self.data[metric] - min_val) / (max_val - min_val)
                
                # 绘制所有指标的趋势
                for metric in metrics_to_normalize:
                    if metric in normalized_data.columns:
                        self.ax_trend.plot(normalized_data['train_progress'], normalized_data[metric], linewidth=2, label=metric.split('/')[-1])
            
            self.ax_trend.set_xlabel('Training Progress (%)')
            self.ax_trend.set_ylabel('Normalized Metrics')
            self.ax_trend.grid(True, alpha=0.3)
            self.ax_trend.legend()
            
        except Exception as e:
            print(f"Failed to update plots: {e}")
    
    def save_current_plots(self):
        """保存当前图表"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.save_dir / f'training_monitor_{timestamp}.png'
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
    
    def start_monitoring(self, save_interval=300):
        """开始监控"""
        print(f"Starting training monitoring...")
        print(f"Monitoring file: {self.results_path}")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Auto save interval: {save_interval} seconds")
        
        if self.headless:
            print("Running in headless mode - charts will be saved as files only")
        else:
            print("Press Ctrl+C to stop monitoring")
        
        last_save = time.time()
        
        try:
            while True:
                if self.load_data():
                    self.update_plots()
                    
                    # 在无头模式下不使用plt.pause
                    if not self.headless:
                        plt.pause(0.1)  # 短暂暂停以更新显示
                    
                    # 定期保存图表
                    if time.time() - last_save > save_interval:
                        self.save_current_plots()
                        last_save = time.time()
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            self.save_current_plots()
            if not self.headless:
                plt.show()
    
    def generate_report(self):
        """生成训练报告"""
        if self.data is None or len(self.data) == 0:
            print("No data available for report generation")
            return
            
        try:
            report_path = self.save_dir / 'training_report.txt'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("YOLOv5 Small Target Detection Training Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Training Epochs: {len(self.data)}\n\n")
                
                if len(self.data) > 0:
                    latest = self.data.iloc[-1]
                    
                    f.write("Training Analysis:\n")
                    f.write("-" * 20 + "\n")
                    
                    # 分析训练状态
                    valid_epochs = 0
                    if 'train/box_loss' in self.data.columns:
                        valid_epochs = self.data['train/box_loss'].notna().sum()
                    
                    f.write(f"Valid training epochs: {valid_epochs} / {len(self.data)}\n")
                    
                    if valid_epochs > 0:
                        # 损失分析
                        f.write("\nLoss Analysis:\n")
                        f.write("-" * 15 + "\n")
                        
                        if 'train/box_loss' in self.data.columns:
                            box_loss_data = self.data['train/box_loss'].dropna()
                            if len(box_loss_data) > 0:
                                initial_box = box_loss_data.iloc[0]
                                final_box = box_loss_data.iloc[-1]
                                min_box = box_loss_data.min()
                                if pd.notna(initial_box) and pd.notna(final_box) and initial_box > 0:
                                    improvement = ((initial_box - final_box) / initial_box * 100)
                                    f.write(f"Box Loss: {initial_box:.4f} -> {final_box:.4f} "
                                           f"(Change: {improvement:+.1f}%, Min: {min_box:.4f})\n")
                        
                        if 'train/obj_loss' in self.data.columns:
                            obj_loss_data = self.data['train/obj_loss'].dropna()
                            if len(obj_loss_data) > 0:
                                initial_obj = obj_loss_data.iloc[0]
                                final_obj = obj_loss_data.iloc[-1]
                                min_obj = obj_loss_data.min()
                                if pd.notna(initial_obj) and pd.notna(final_obj) and initial_obj > 0:
                                    improvement = ((initial_obj - final_obj) / initial_obj * 100)
                                    f.write(f"Obj Loss: {initial_obj:.4f} -> {final_obj:.4f} "
                                           f"(Change: {improvement:+.1f}%, Min: {min_obj:.4f})\n")
                        
                        if 'train/cls_loss' in self.data.columns:
                            cls_loss_data = self.data['train/cls_loss'].dropna()
                            if len(cls_loss_data) > 0:
                                # 检查是否为分类任务
                                if cls_loss_data.max() > 0:
                                    initial_cls = cls_loss_data.iloc[0]
                                    final_cls = cls_loss_data.iloc[-1]
                                    min_cls = cls_loss_data.min()
                                    f.write(f"Cls Loss: {initial_cls:.4f} -> {final_cls:.4f} "
                                           f"(Min: {min_cls:.4f})\n")
                                else:
                                    f.write("Cls Loss: 0.0000 (No classification loss - detection only)\n")
                    
                    # 验证指标分析
                    f.write("\nValidation Metrics:\n")
                    f.write("-" * 18 + "\n")
                    
                    metrics_available = False
                    for metric in ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'metrics/precision', 'metrics/recall']:
                        if metric in self.data.columns:
                            metric_data = self.data[metric].dropna()
                            if len(metric_data) > 0 and metric_data.max() > 0:
                                metrics_available = True
                                final_val = metric_data.iloc[-1] if pd.notna(metric_data.iloc[-1]) else 0
                                max_val = metric_data.max()
                                best_epoch = metric_data.idxmax() + 1
                                metric_name = metric.split('/')[-1]
                                f.write(f"{metric_name}: {final_val:.4f} (Best: {max_val:.4f} at epoch {best_epoch})\n")
                    
                    if not metrics_available:
                        f.write("⚠️  No validation metrics available!\n")
                        f.write("This indicates that validation was not performed during training.\n")
                        f.write("Possible causes:\n")
                        f.write("- No validation dataset provided\n")
                        f.write("- Validation failed due to data issues\n")
                        f.write("- Training stopped before validation phase\n")
                    
                    # 学习率分析
                    f.write("\nLearning Rate:\n")
                    f.write("-" * 14 + "\n")
                    lr_col = None
                    if 'x/lr0' in self.data.columns:
                        lr_col = 'x/lr0'
                    elif 'lr/pg0' in self.data.columns:
                        lr_col = 'lr/pg0'
                    
                    if lr_col:
                        lr_data = self.data[lr_col].dropna()
                        if len(lr_data) > 0:
                            initial_lr = lr_data.iloc[0]
                            final_lr = lr_data.iloc[-1]
                            f.write(f"Initial LR: {initial_lr:.6f}\n")
                            f.write(f"Final LR: {final_lr:.6f}\n")
                    
                    # 训练问题诊断
                    f.write("\nTraining Issues Detected:\n")
                    f.write("-" * 26 + "\n")
                    
                    issues_found = False
                    
                    # 检查NaN值
                    nan_epochs = []
                    if 'train/box_loss' in self.data.columns:
                        nan_mask = self.data['train/box_loss'].isna()
                        if nan_mask.any():
                            nan_epochs = self.data[nan_mask].index.tolist()
                            issues_found = True
                            f.write(f"⚠️  Training failed at epochs: {nan_epochs[:10]}{'...' if len(nan_epochs) > 10 else ''}\n")
                    
                    # 检查验证缺失
                    if not metrics_available:
                        issues_found = True
                        f.write("⚠️  No validation metrics computed\n")
                    
                    # 检查损失趋势
                    if 'train/box_loss' in self.data.columns:
                        box_loss_data = self.data['train/box_loss'].dropna()
                        if len(box_loss_data) > 10:
                            recent_loss = box_loss_data.tail(10).mean()
                            early_loss = box_loss_data.head(10).mean()
                            if recent_loss > early_loss * 0.9:  # 损失没有明显下降
                                issues_found = True
                                f.write("⚠️  Training loss not decreasing significantly\n")
                    
                    if not issues_found:
                        f.write("✓ No major issues detected\n")
                    
                    # 建议
                    f.write("\nRecommendations:\n")
                    f.write("-" * 15 + "\n")
                    
                    if nan_epochs:
                        f.write("• Fix training instability (reduce batch size, lower learning rate)\n")
                    if not metrics_available:
                        f.write("• Add validation dataset for proper model evaluation\n")
                        f.write("• Check dataset paths and format\n")
                    if valid_epochs < len(self.data) * 0.8:
                        f.write("• Investigate training failures (memory, data loading)\n")
                    
                    f.write("• Consider reducing batch size if GPU memory issues occur\n")
                    f.write("• Monitor training with validation data for better results\n")
            
            print(f"Training report saved to: {report_path}")
            
        except Exception as e:
            print(f"Failed to generate report: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv5训练监控工具')
    parser.add_argument('--results', type=str, required=True, 
                       help='results.csv文件路径')
    parser.add_argument('--save-dir', type=str, 
                       help='图表保存目录 (默认为results.csv同目录)')
    parser.add_argument('--update-interval', type=int, default=30,
                       help='更新间隔(秒) (默认: 30)')
    parser.add_argument('--save-interval', type=int, default=300,
                       help='自动保存间隔(秒) (默认: 300)')
    parser.add_argument('--report-only', action='store_true',
                       help='仅生成报告，不启动实时监控')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = EnhancedTrainingMonitor(
        results_path=args.results,
        save_dir=args.save_dir,
        update_interval=args.update_interval
    )
    
    if args.report_only:
        # 仅生成报告
        if monitor.load_data():
            monitor.update_plots()
            monitor.save_current_plots()
            monitor.generate_report()
        else:
            print("无法加载数据文件")
    else:
        # 启动实时监控
        monitor.start_monitoring(save_interval=args.save_interval)


if __name__ == '__main__':
    main()