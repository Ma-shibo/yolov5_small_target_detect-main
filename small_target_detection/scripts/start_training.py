#!/bin/bash
# -*- coding: utf-8 -*-
"""
YOLOv5 小目标检测训练启动脚本
使用优化的参数配置，便于监控和调节超参数
支持断点继续训练功能和定时训练功能
"""

import os
import sys
from pathlib import Path

# 添加YOLOv5路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 向上两级到yolov5-v7目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import subprocess
import argparse
import time
import threading
import glob
import yaml
import signal
import datetime
import json

def find_latest_checkpoint(save_dir):
    """查找最新的检查点文件"""
    weights_dir = Path(save_dir) / "weights"
    if not weights_dir.exists():
        return None, None
    
    # 查找last.pt文件（最新检查点）
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        return str(last_pt), "last"
    
    # 查找最新的epoch检查点
    epoch_files = list(weights_dir.glob("epoch*.pt"))
    if epoch_files:
        # 按修改时间排序，获取最新的
        latest_epoch = max(epoch_files, key=lambda x: x.stat().st_mtime)
        epoch_num = latest_epoch.stem.replace("epoch", "")
        return str(latest_epoch), f"epoch{epoch_num}"
    
    return None, None

def get_training_status(save_dir):
    """获取训练状态信息"""
    results_file = Path(save_dir) / "results.csv"
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # 有数据行
                last_line = lines[-1].strip()
                parts = last_line.split(',')
                if len(parts) > 0:
                    epoch = int(float(parts[0])) + 1  # 下一个epoch
                    return {
                        'last_epoch': epoch - 1,
                        'next_epoch': epoch,
                        'total_lines': len(lines) - 1
                    }
    except Exception as e:
        print(f"⚠️  读取训练状态失败: {e}")
    
    return None

def check_resume_training(config):
    """检查是否可以继续训练"""
    save_dir = Path(config['project']) / config['name']
    
    if not save_dir.exists():
        return False, None, None
    
    # 查找检查点
    checkpoint_path, checkpoint_type = find_latest_checkpoint(save_dir)
    if not checkpoint_path:
        return False, None, None
    
    # 获取训练状态
    status = get_training_status(save_dir)
    
    return True, checkpoint_path, status

def prompt_resume_choice():
    """提示用户选择继续训练方式"""
    print("\n🔄 检测到已有训练记录，请选择:")
    print("1. 从断点继续训练 (推荐)")
    print("2. 重新开始训练")
    print("3. 取消训练")
    
    while True:
        choice = input("\n请输入选择 (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("❌ 无效选择，请输入 1、2 或 3")

def parse_time_duration(time_str):
    """解析时间字符串，支持多种格式
    支持格式：
    - 数字+单位: 2h, 30m, 1.5h, 90m
    - 时:分格式: 2:30, 1:45
    - 纯数字（分钟）: 120
    """
    if not time_str:
        return None
    
    time_str = time_str.lower().strip()
    
    try:
        # 时:分格式 (如 2:30)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                hours = float(parts[0])
                minutes = float(parts[1])
                return int((hours * 60 + minutes) * 60)  # 转换为秒
        
        # 带单位格式 (如 2h, 30m, 1.5h)
        elif time_str.endswith('h'):
            hours = float(time_str[:-1])
            return int(hours * 3600)  # 转换为秒
        elif time_str.endswith('m'):
            minutes = float(time_str[:-1])
            return int(minutes * 60)  # 转换为秒
        elif time_str.endswith('s'):
            seconds = float(time_str[:-1])
            return int(seconds)
        
        # 纯数字（默认为分钟）
        else:
            minutes = float(time_str)
            return int(minutes * 60)  # 转换为秒
            
    except ValueError:
        raise ValueError(f"无效的时间格式: {time_str}")

def format_duration(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}分{secs}秒" if secs > 0 else f"{minutes}分"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if minutes > 0:
            return f"{hours}小时{minutes}分"
        else:
            return f"{hours}小时"

def save_training_session_info(save_dir, config, start_time, planned_duration):
    """保存训练会话信息"""
    # 确保目录存在
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    session_info = {
        'start_time': start_time.isoformat(),
        'planned_duration_seconds': planned_duration,
        'config': config,
        'session_id': f"session_{int(start_time.timestamp())}"
    }
    
    session_file = save_dir / 'training_session.json'
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_info, f, indent=2, ensure_ascii=False)
    
    return session_file

def load_training_session_info(save_dir):
    """加载训练会话信息"""
    session_file = Path(save_dir) / 'training_session.json'
    if session_file.exists():
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  读取会话信息失败: {e}")
    return None

class TimedTrainingController:
    """定时训练控制器"""
    
    def __init__(self, duration_seconds, save_dir):
        self.duration_seconds = duration_seconds
        self.save_dir = Path(save_dir)
        self.start_time = None
        self.process = None
        self.timer_thread = None
        self.is_stopped = False
        
    def start_timer(self, process):
        """启动计时器"""
        self.process = process
        self.start_time = datetime.datetime.now()
        
        print(f"⏱️  定时训练已启动，计划运行时间: {format_duration(self.duration_seconds)}")
        print(f"🕐 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 启动计时器线程
        self.timer_thread = threading.Thread(target=self._timer_worker)
        self.timer_thread.daemon = True
        self.timer_thread.start()
        
        # 启动进度显示线程
        progress_thread = threading.Thread(target=self._progress_worker)
        progress_thread.daemon = True
        progress_thread.start()
    
    def _timer_worker(self):
        """计时器工作线程"""
        time.sleep(self.duration_seconds)
        if not self.is_stopped and self.process:
            print(f"\n⏰ 定时训练时间到达({format_duration(self.duration_seconds)})，正在安全停止训练...")
            self._stop_training()
    
    def _progress_worker(self):
        """进度显示工作线程"""
        while not self.is_stopped and self.process and self.process.poll() is None:
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            if elapsed < self.duration_seconds:
                remaining = self.duration_seconds - elapsed
                progress = (elapsed / self.duration_seconds) * 100
                
                # 每5分钟显示一次进度
                if int(elapsed) % 300 == 0 and int(elapsed) > 0:
                    print(f"⏳ 训练进度: {progress:.1f}% | 剩余时间: {format_duration(int(remaining))}")
            
            time.sleep(60)  # 每分钟检查一次
    
    def _stop_training(self):
        """安全停止训练"""
        if self.process and self.process.poll() is None:
            self.is_stopped = True
            
            # 发送SIGINT信号（相当于Ctrl+C）
            try:
                self.process.send_signal(signal.SIGINT)
                print("📤 已发送停止信号...")
                
                # 等待进程优雅退出
                try:
                    self.process.wait(timeout=30)
                    print("✅ 训练已安全停止")
                except subprocess.TimeoutExpired:
                    print("⚠️  等待超时，强制终止进程...")
                    self.process.terminate()
                    self.process.wait(timeout=10)
                
            except Exception as e:
                print(f"❌ 停止训练时出错: {e}")
                self.process.terminate()
    
    def get_elapsed_time(self):
        """获取已用时间"""
        if self.start_time:
            return (datetime.datetime.now() - self.start_time).total_seconds()
        return 0

def run_training(config):
    """运行训练"""
    save_dir = Path(config['project']) / config['name']
    
    # 检查是否可以继续训练
    can_resume, checkpoint_path, status = check_resume_training(config)
    
    resume_training = False
    if can_resume and not config['force_restart']:
        print(f"\n📁 发现训练目录: {save_dir}")
        print(f"📊 检查点文件: {checkpoint_path}")
        
        if status:
            print(f"📈 训练进度: 已完成 {status['last_epoch']} 轮，下一轮: {status['next_epoch']}")
        
        # 显示之前的训练会话信息
        session_info = load_training_session_info(save_dir)
        if session_info:
            prev_start = datetime.datetime.fromisoformat(session_info['start_time'])
            prev_duration = session_info.get('planned_duration_seconds', 0)
            print(f"📅 上次训练: {prev_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  上次计划时长: {format_duration(prev_duration)}")
        
        if config['auto_resume']:
            print("🔄 自动从断点继续训练...")
            resume_training = True
        else:
            choice = prompt_resume_choice()
            if choice == '1':
                resume_training = True
            elif choice == '2':
                print("🔥 将重新开始训练...")
            else:
                print("❌ 训练已取消")
                return False
    
    print("🚀 开始YOLOv5小目标检测训练...")
    print("=" * 60)
    
    # 保存训练会话信息
    if config.get('time_limit'):
        duration_seconds = parse_time_duration(config['time_limit'])
        save_training_session_info(save_dir, config, datetime.datetime.now(), duration_seconds)
        print(f"📝 已保存训练会话信息，计划运行时间: {format_duration(duration_seconds)}")
    
    # 构建训练命令
    train_cmd = [
        sys.executable, "small_target_train.py",
        "--data", config['data'],
        "--cfg", config['cfg'], 
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--imgsz", str(config['imgsz']),
        "--project", config['project'],
        "--name", config['name'],
        "--hyp", config['hyp'],
        "--optimizer", config['optimizer'],
        "--cos-lr",  # 启用余弦学习率
        "--save-period", str(config['save_period']),
        "--patience", str(config['patience']),
        "--label-smoothing", str(config['label_smoothing']),
        "--device", config['device']
    ]
    
    # 添加稳定性参数
    if config.get('fix_instability', False):
        train_cmd.append("--fix-instability")
        train_cmd.extend(["--stability-level", config.get('stability_level', 'moderate')])
        train_cmd.extend(["--gradient-clip-norm", str(config.get('gradient_clip_norm', 1.0))])
        train_cmd.extend(["--nan-detection-window", str(config.get('nan_detection_window', 3))])
    
    # 设置权重参数
    if resume_training and checkpoint_path:
        train_cmd.extend(["--weights", checkpoint_path])
        print(f"🔄 从检查点继续训练: {checkpoint_path}")
    else:
        train_cmd.extend(["--weights", config['weights']])
        print(f"🎯 使用预训练权重: {config['weights']}")
    
    # 添加可选参数
    if config['multi_scale']:
        train_cmd.append("--multi-scale")
    if config['image_weights']:
        train_cmd.append("--image-weights")
    if config['exist_ok'] or resume_training:
        train_cmd.append("--exist-ok")
        
    print("训练命令:")
    print(" ".join(train_cmd))
    print("=" * 60)
    
    # 初始化定时控制器
    timer_controller = None
    if config.get('time_limit'):
        try:
            duration_seconds = parse_time_duration(config['time_limit'])
            timer_controller = TimedTrainingController(duration_seconds, save_dir)
        except ValueError as e:
            print(f"❌ 时间格式错误: {e}")
            return False
    
    # 启动训练
    try:
        process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, universal_newlines=True)
        
        # 启动定时器（如果设置了时间限制）
        if timer_controller:
            timer_controller.start_timer(process)
        
        # 实时输出训练日志
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
        process.wait()
        
        # 输出训练结果
        if timer_controller:
            elapsed_time = timer_controller.get_elapsed_time()
            print(f"\n⏱️  实际训练时间: {format_duration(int(elapsed_time))}")
            
            if timer_controller.is_stopped:
                print("🔄 训练已按计划时间停止，检查点已保存，可稍后继续训练")
                return True  # 定时停止也算成功
        
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        if timer_controller:
            timer_controller._stop_training()
        else:
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ 训练启动失败: {e}")
        return False

def start_monitor(results_path, save_dir):
    """启动训练监控"""
    monitor_cmd = [
        sys.executable, "training_monitor.py",
        "--results", str(results_path),
        "--save-dir", str(save_dir),
        "--update-interval", "30",
        "--save-interval", "300"
    ]
    
    print("🖥️  启动训练监控...")
    try:
        subprocess.Popen(monitor_cmd)
        print("✅ 监控已启动，请查看弹出的图表窗口")
    except Exception as e:
        print(f"⚠️  监控启动失败: {e}")

def check_dataset(data_path):
    """检查数据集配置"""
    if not Path(data_path).exists():
        print(f"❌ 数据集配置文件不存在: {data_path}")
        return False
    
    print(f"✅ 数据集配置: {data_path}")
    return True

def check_weights(weights_path):
    """检查预训练权重"""
    if not Path(weights_path).exists():
        print(f"⚠️  预训练权重不存在: {weights_path}")
        print("将尝试自动下载...")
        return True  # YOLOv5会自动下载
    
    print(f"✅ 预训练权重: {weights_path}")
    return True

def create_config():
    """创建训练配置"""
    parser = argparse.ArgumentParser(description='YOLOv5小目标检测训练启动器')
    
    # 基础配置
    parser.add_argument('--data', type=str, 
                       default='/home/msb/pan1/A.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--cfg', type=str, default='../../models/yolov5s.yaml',
                       help='模型配置文件')
    parser.add_argument('--weights', type=str, default='../../weights/yolov5s.pt',
                       help='预训练权重路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=140,
                       help='训练轮次 (推荐: 200-500)')
    parser.add_argument('--batch-size', type=int, default=14,
                       help='批次大小 (根据GPU内存调整)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU设备 (0, 1, 2... 或 cpu)')
    
    # 优化器和学习率
    parser.add_argument('--optimizer', type=str, default='AdamW',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='优化器选择 (推荐: AdamW)')
    parser.add_argument('--hyp', type=str, 
                       default='../../data/hyps/hyp.small-target-conservative.yaml',
                       help='超参数文件')
    
    # 保存和监控
    parser.add_argument('--project', type=str, default='runs/small_target_train',
                       help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp',
                       help='实验名称')
    parser.add_argument('--train-dir', type=str, default=None,
                       help='直接指定训练目录路径（优先于project和name组合）')
    parser.add_argument('--save-period', type=int, default=10,
                       help='模型保存间隔 (推荐: 5-20)')
    parser.add_argument('--patience', type=int, default=50,
                       help='早停耐心度 (推荐: 30-100)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='标签平滑 (推荐: 0.0-0.2)')
    
    # 断点继续训练参数
    parser.add_argument('--auto-resume', action='store_true',
                       help='自动从断点继续训练 (不询问)')
    parser.add_argument('--force-restart', action='store_true',
                       help='强制重新开始训练 (忽略检查点)')
    
    # 定时训练参数
    parser.add_argument('--time-limit', type=str,
                       help='训练时间限制 (格式: 2h, 30m, 2:30, 120)')
    parser.add_argument('--schedule-training', action='store_true',
                       help='启用定时训练模式 (显示更多时间信息)')
    
    # 增强选项
    parser.add_argument('--multi-scale', action='store_true', default=True,
                       help='启用多尺度训练 (需要更多显存)')
    parser.add_argument('--image-weights', action='store_true',
                       help='启用图像权重采样')
    parser.add_argument('--exist-ok', action='store_true',
                       help='允许覆盖现有实验')
    
    # 增强稳定性选项
    parser.add_argument('--fix-instability', action='store_true',
                       help='应用训练稳定性修复')
    parser.add_argument('--stability-level', type=str, default='moderate',
                       choices=['light', 'moderate', 'aggressive', 'extreme'],
                       help='稳定性修复程度（light=轻微, moderate=中度, aggressive=激进, extreme=极限）')
    parser.add_argument('--nan-detection-window', type=int, default=3,
                       help='NaN检测窗口（连续出现NaN的批次数达到此值时采取措施）')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0,
                       help='梯度裁剪范数值（更小的值训练更稳定但可能收敛更慢）')
    
    # 监控选项
    parser.add_argument('--monitor', action='store_true', default=False,
                       help='启用训练监控')
    parser.add_argument('--no-monitor', dest='monitor', action='store_false',
                       help='禁用训练监控')
    
    args = parser.parse_args()
    return vars(args)

def print_config(config):
    """打印配置信息"""
    print("📋 训练配置:")
    print("-" * 40)
    print(f"数据集: {config['data']}")
    print(f"模型: {config['cfg']}")
    print(f"预训练权重: {config['weights']}")
    print(f"训练轮次: {config['epochs']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"图像尺寸: {config['imgsz']}")
    print(f"设备: {config['device']}")
    print(f"优化器: {config['optimizer']}")
    print(f"超参数文件: {config['hyp']}")
    print(f"保存目录: {config['project']}/{config['name']}")
    print(f"保存间隔: 每{config['save_period']}轮")
    print(f"早停耐心度: {config['patience']}轮")
    print(f"多尺度训练: {'✅' if config['multi_scale'] else '❌'}")
    print(f"图像权重: {'✅' if config['image_weights'] else '❌'}")
    print(f"训练监控: {'✅' if config['monitor'] else '❌'}")
    print(f"自动断点继续: {'✅' if config['auto_resume'] else '❌'}")
    print(f"强制重新训练: {'✅' if config['force_restart'] else '❌'}")
    print(f"训练稳定性修复: {'✅' if config.get('fix_instability', False) else '❌'}")
    if config.get('fix_instability', False):
        print(f"- 稳定性级别: {config.get('stability_level', 'moderate')}")
        print(f"- 梯度裁剪范数: {config.get('gradient_clip_norm', 1.0)}")
        print(f"- NaN检测窗口: {config.get('nan_detection_window', 3)}")
    
    # 显示定时训练信息
    if config.get('time_limit'):
        try:
            duration_seconds = parse_time_duration(config['time_limit'])
            print(f"⏱️  定时训练: {format_duration(duration_seconds)}")
        except ValueError as e:
            print(f"❌ 时间格式错误: {e}")
    
    print("-" * 40)

def main():
    """主函数"""
    print("🎯 YOLOv5 小目标检测训练启动器")
    print("=" * 60)
    
    # 获取配置
    config = create_config()
    
    # 打印配置
    print_config(config)
    
    # 检查必要文件
    if not check_dataset(config['data']):
        return
    
    if not check_weights(config['weights']):
        return
    
    # 确认开始训练
    print("\n🤔 确认开始训练? (y/N): ", end="")
    confirm = input().lower().strip()
    
    if confirm not in ['y', 'yes', '是']:
        print("❌ 训练已取消")
        return
    
    # 创建结果路径
    results_path = Path(config['project']) / config['name'] / 'results.csv'
    save_dir = Path(config['project']) / config['name']
    
    # 启动监控 (如果启用)
    if config['monitor']:
        def delayed_monitor():
            time.sleep(30)  # 等待30秒后启动监控
            if results_path.exists():
                start_monitor(results_path, save_dir)
        
        monitor_thread = threading.Thread(target=delayed_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    # 开始训练
    success = run_training(config)
    
    if success:
        print("\n🎉 训练完成!")
        print(f"📊 结果保存在: {save_dir}")
        print(f"📈 查看结果文件: {results_path}")
        
        # 生成最终报告
        if results_path.exists():
            report_cmd = [
                sys.executable, "training_monitor.py",
                "--results", str(results_path),
                "--report-only"
            ]
            subprocess.run(report_cmd)
    else:
        print("\n❌ 训练失败或被中断")

if __name__ == '__main__':
    main()