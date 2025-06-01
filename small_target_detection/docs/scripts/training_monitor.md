# training_monitor.py - 训练监控脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/training_monitor.py`

## 📋 功能概述

实时监控YOLOv5训练进程的可视化工具，提供训练状态追踪、性能分析和异常检测功能。

### 主要特性
- 📊 实时训练指标监控
- 🔍 异常检测和报警
- 📈 动态图表更新
- 💾 训练日志分析
- 🎯 性能瓶颈诊断

## 🚀 使用方法

### 基础监控
```bash
# 监控指定训练目录
python scripts/training_monitor.py --logdir runs/train/exp

# 实时监控当前训练
python scripts/training_monitor.py --auto-detect

# 监控多个实验
python scripts/training_monitor.py --logdir runs/train --recursive
```

### 高级监控
```bash
# 启用告警功能
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --alert \
    --email your@email.com \
    --threshold-loss 0.1 \
    --threshold-map 0.5

# 自定义刷新间隔
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --refresh-interval 30 \
    --port 8080
```

## 📝 参数详解

### 基础参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--logdir` | str | 无 | 训练日志目录路径 | 命令行 |
| `--auto-detect` | bool | False | 自动检测最新训练 | 命令行 |
| `--recursive` | bool | False | 递归监控子目录 | 命令行 |

### 显示配置
| 参数 | 类型 | 范围 | 默认值 | 说明 | 修改位置 |
|------|------|------|--------|------|----------|
| `--refresh-interval` | int | 5-300 | 10 | 刷新间隔(秒) | 命令行 |
| `--port` | int | 1000-9999 | 8888 | Web界面端口 | 命令行 |
| `--host` | str | - | localhost | 服务器地址 | 命令行 |
| `--theme` | str | light/dark | light | 界面主题 | 命令行 |

### 告警配置
| 参数 | 类型 | 范围 | 默认值 | 说明 | 修改位置 |
|------|------|------|--------|------|----------|
| `--alert` | bool | - | False | 启用告警功能 | 命令行 |
| `--email` | str | - | 无 | 告警邮箱地址 | 命令行 |
| `--threshold-loss` | float | 0.01-10.0 | 1.0 | 损失告警阈值 | 命令行 |
| `--threshold-map` | float | 0.0-1.0 | 0.3 | mAP告警阈值 | 命令行 |
| `--patience` | int | 5-100 | 20 | 性能停滞容忍轮次 | 命令行 |

### 分析配置
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--save-plots` | bool | False | 保存监控图表 | 命令行 |
| `--export-data` | bool | False | 导出监控数据 | 命令行 |
| `--compare-mode` | bool | False | 多实验对比模式 | 命令行 |

## 📊 监控指标说明

### 1. 训练损失指标
- **train/box_loss**: 边界框回归损失
- **train/obj_loss**: 目标性检测损失  
- **train/cls_loss**: 分类损失
- **train/total_loss**: 总训练损失

### 2. 验证性能指标
- **val/box_loss**: 验证边界框损失
- **val/obj_loss**: 验证目标性损失
- **val/cls_loss**: 验证分类损失
- **metrics/precision**: 精确度
- **metrics/recall**: 召回率
- **metrics/mAP_0.5**: mAP@0.5
- **metrics/mAP_0.5:0.95**: mAP@0.5:0.95

### 3. 学习率监控
- **lr/pg0**: 骨干网络学习率
- **lr/pg1**: 颈部网络学习率
- **lr/pg2**: 检测头学习率

### 4. 系统监控指标
- **GPU利用率**: GPU使用百分比
- **显存使用**: 显存占用情况
- **训练速度**: 每秒处理图片数
- **ETA**: 预计完成时间

## 🖥️ Web界面功能

### 主控制台
访问 `http://localhost:8888` 查看训练监控界面

#### 1. 实时图表区域
```
损失曲线图表:
├── 训练损失 (train_loss)
├── 验证损失 (val_loss)  
├── 总损失对比 (total_loss)
└── 损失组件分解 (loss_breakdown)

性能指标图表:
├── mAP趋势 (map_trend)
├── 精确度召回率 (precision_recall)
├── F1分数 (f1_score)
└── 类别AP对比 (class_ap)

学习率图表:
├── 学习率调度 (lr_schedule)
├── 多组学习率 (lr_groups)
└── 学习率vs性能 (lr_vs_performance)
```

#### 2. 状态监控面板
```
训练状态:
├── 当前轮次/总轮次
├── 已训练时间
├── 预计剩余时间
├── 平均每轮用时
└── 训练进度百分比

硬件状态:
├── GPU温度
├── GPU利用率
├── 显存使用率
├── CPU使用率
└── 内存使用率

最新指标:
├── 当前损失值
├── 最佳mAP
├── 最新精确度
├── 最新召回率
└── F1分数
```

#### 3. 控制操作区
```
训练控制:
├── 暂停/恢复训练
├── 提前停止
├── 保存检查点
└── 调整学习率

数据导出:
├── 导出训练数据
├── 保存当前图表
├── 生成训练报告
└── 下载最佳模型
```

### 对比分析页面
访问 `http://localhost:8888/compare` 进行多实验对比

```
实验选择:
├── 选择对比实验
├── 设置对比指标
├── 选择时间范围
└── 应用过滤条件

对比图表:
├── 多实验损失对比
├── 性能指标对比
├── 训练效率对比
└── 收敛速度分析

统计分析:
├── 最终性能排名
├── 收敛时间统计
├── 稳定性分析
└── 资源消耗对比
```

## 📊 输出文件说明

### 监控数据目录结构
```
monitoring_output/
├── real_time_data/              # 实时监控数据
│   ├── training_metrics.json   # 训练指标数据
│   ├── system_stats.json       # 系统状态数据
│   └── alerts.log              # 告警日志
├── plots/                       # 监控图表
│   ├── loss_curves.png         # 损失曲线图
│   ├── performance_metrics.png # 性能指标图
│   ├── learning_rate.png       # 学习率图
│   └── system_monitoring.png   # 系统监控图
├── reports/                     # 监控报告
│   ├── training_summary.html   # 训练总结报告
│   ├── performance_analysis.pdf # 性能分析报告
│   └── comparison_report.xlsx  # 对比分析报告
└── exports/                     # 导出数据
    ├── metrics_export.csv      # 指标数据导出
    ├── alerts_summary.txt      # 告警汇总
    └── training_config.yaml    # 训练配置备份
```

### 主要输出文件解释

#### 1. 实时监控数据 (training_metrics.json)
```json
{
  "timestamp": "2024-01-15 10:30:00",
  "epoch": 50,
  "training_metrics": {
    "train_loss": 0.0234,
    "train_box_loss": 0.0156,
    "train_obj_loss": 0.0067,
    "train_cls_loss": 0.0011
  },
  "validation_metrics": {
    "val_loss": 0.0198,
    "precision": 0.856,
    "recall": 0.743,
    "mAP_50": 0.798,
    "mAP_50_95": 0.567
  },
  "learning_rates": {
    "pg0": 0.00834,
    "pg1": 0.00834,
    "pg2": 0.00834
  },
  "system_stats": {
    "gpu_utilization": 98.5,
    "gpu_memory": 85.2,
    "gpu_temperature": 76,
    "training_speed": 245.6
  }
}
```

#### 2. 告警日志 (alerts.log)
```
[2024-01-15 10:15:23] WARNING: Training loss increased for 5 consecutive epochs
[2024-01-15 10:20:45] ALERT: GPU temperature exceeded 80°C (current: 82°C)
[2024-01-15 10:25:12] INFO: mAP@0.5 reached new best: 0.798
[2024-01-15 10:30:30] WARNING: No improvement in mAP for 20 epochs
```

#### 3. 训练总结报告 (training_summary.html)
包含完整的训练过程分析：
- 训练配置总结
- 性能指标趋势
- 收敛分析
- 异常检测结果
- 资源使用统计
- 改进建议

## ⚠️ 告警机制

### 1. 性能异常告警
```python
# 可配置的告警条件
ALERT_CONDITIONS = {
    'loss_increase': {
        'condition': 'train_loss连续5轮上升',
        'action': '发送邮件+日志记录'
    },
    'no_improvement': {
        'condition': 'mAP超过patience轮无改善',
        'action': '建议提前停止'
    },
    'overfitting': {
        'condition': 'val_loss/train_loss > 1.5',
        'action': '过拟合警告'
    }
}
```

### 2. 系统资源告警
```python
SYSTEM_ALERTS = {
    'gpu_temperature': {
        'threshold': 80,  # 摄氏度
        'action': '降低训练强度'
    },
    'gpu_memory': {
        'threshold': 95,  # 百分比
        'action': '减少批次大小'
    },
    'disk_space': {
        'threshold': 90,  # 百分比
        'action': '清理临时文件'
    }
}
```

### 3. 邮件告警配置
```bash
# 配置SMTP设置
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export EMAIL_USER="your_email@gmail.com"
export EMAIL_PASS="your_app_password"

# 启用邮件告警
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --alert \
    --email recipient@email.com \
    --email-interval 300  # 最小邮件间隔(秒)
```

## 🔧 高级功能

### 1. 自定义监控指标
```python
# 添加自定义指标监控
CUSTOM_METRICS = {
    'convergence_rate': {
        'formula': 'delta_map / delta_epoch',
        'description': '收敛速度'
    },
    'efficiency_score': {
        'formula': 'map_50 / training_time',
        'description': '训练效率'
    },
    'stability_index': {
        'formula': 'std(last_10_epochs_map)',
        'description': '训练稳定性'
    }
}
```

### 2. 分布式训练监控
```bash
# 监控多GPU分布式训练
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --distributed \
    --gpu-count 4 \
    --show-per-gpu-stats
```

### 3. 云端监控部署
```bash
# 部署到云端服务器
python scripts/training_monitor.py \
    --logdir /remote/training/logs \
    --host 0.0.0.0 \
    --port 8888 \
    --ssl-cert /path/to/cert.pem \
    --ssl-key /path/to/key.pem
```

### 4. API接口
```python
# REST API接口示例
GET /api/metrics/latest          # 获取最新指标
GET /api/training/status         # 获取训练状态
POST /api/training/control       # 控制训练进程
GET /api/alerts/history          # 获取告警历史
```

## 📱 移动端支持

### 响应式Web界面
- 📱 手机浏览器兼容
- 📊 触控友好的图表交互
- 🔔 推送通知支持
- 📶 离线数据缓存

### 移动端功能
```
主要功能:
├── 实时指标查看
├── 告警接收
├── 训练状态查询
├── 简单控制操作
└── 图表截图分享
```

## 🔗 相关文档

- [small_target_train.md](small_target_train.md) - 训练脚本使用
- [evaluate_detection.md](evaluate_detection.md) - 评估分析
- [generate_charts.md](generate_charts.md) - 图表生成
- [gui_test.md](gui_test.md) - 图形界面测试