# start_training.py - 训练启动脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/start_training.py`

## 📋 功能概述

这是一个功能强大的YOLOv5小目标检测训练启动器，提供智能断点恢复、定时训练、实时监控等高级功能，让训练过程更加便捷和可控。

### 主要特性
- 🔄 智能断点恢复功能
- ⏱️ 定时训练控制
- 📊 实时训练监控集成
- 🎯 小目标检测专用优化
- 💾 训练会话信息管理
- 🔧 灵活的参数配置

## 🚀 使用方法

### 基础使用
```bash
# 使用默认配置启动训练
python scripts/start_training.py

# 指定数据集和基本参数
python scripts/start_training.py \
    --data data/dataset.yaml \
    --epochs 200 \
    --batch-size 16
```

### 断点恢复训练
```bash
# 自动从断点继续训练（不询问）
python scripts/start_training.py \
    --data data/dataset.yaml \
    --auto-resume

# 强制重新开始训练（忽略检查点）
python scripts/start_training.py \
    --data data/dataset.yaml \
    --force-restart

# 从指定检查点继续训练
python scripts/start_training.py \
    --data data/dataset.yaml \
    --weights runs/small_target_train/exp/weights/last.pt

# 从特定轮次继续训练
python scripts/start_training.py \
    --data data/dataset.yaml \
    --weights runs/small_target_train/exp/weights/epoch50.pt
```

### 定时训练
```bash
# 训练2小时后自动停止
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 2h

# 训练90分钟后自动停止
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 90m

# 训练2小时30分钟后自动停止
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 2:30
```

### 高级配置
```bash
# 完整配置训练
python scripts/start_training.py \
    --data data/small_target.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --epochs 300 \
    --batch-size 32 \
    --imgsz 832 \
    --device 0,1 \
    --optimizer AdamW \
    --hyp data/hyps/hyp.small-target.yaml \
    --project runs/my_experiment \
    --name small_target_v1 \
    --time-limit 4h \
    --auto-resume \
    --monitor
```

## 📝 参数详解

### 基础训练参数
- `--data`: 数据集配置文件路径 (默认: `../../data/dataset.yaml`)
- `--cfg`: 模型配置文件 (默认: `../../models/yolov5s.yaml`)
- `--weights`: 预训练权重路径 (默认: `../../weights/yolov5s.pt`)
- `--epochs`: 训练轮次 (默认: 200, 推荐: 200-500)
- `--batch-size`: 批次大小 (默认: 6, 根据GPU内存调整)
- `--imgsz`: 图像尺寸 (默认: 640)
- `--device`: GPU设备 (默认: '0', 可选: '0', '1', '0,1', 'cpu')

### 优化器参数
- `--optimizer`: 优化器选择 (默认: 'AdamW', 可选: 'SGD', 'Adam', 'AdamW')
- `--hyp`: 超参数文件 (默认: `../../data/hyps/hyp.small-target-conservative.yaml`)

### 保存和监控参数
- `--project`: 项目保存目录 (默认: 'runs/small_target_train')
- `--name`: 实验名称 (默认: 'exp')
- `--save-period`: 模型保存间隔 (默认: 10, 推荐: 5-20)
- `--patience`: 早停耐心度 (默认: 50, 推荐: 30-100)
- `--label-smoothing`: 标签平滑 (默认: 0.1, 推荐: 0.0-0.2)

### 断点恢复参数
- `--auto-resume`: 自动从断点继续训练，不询问用户
- `--force-restart`: 强制重新开始训练，忽略检查点

### 定时训练参数
- `--time-limit`: 训练时间限制，支持多种格式：
  - 数字+单位: `2h`, `30m`, `1.5h`, `90m`
  - 时:分格式: `2:30`, `1:45`
  - 纯数字（分钟）: `120`
- `--schedule-training`: 启用定时训练模式，显示更多时间信息

### 增强选项
- `--multi-scale`: 启用多尺度训练 (默认: True, 需要更多显存)
- `--image-weights`: 启用图像权重采样
- `--exist-ok`: 允许覆盖现有实验

### 监控选项
- `--monitor`: 启用训练监控 (默认: True)
- `--no-monitor`: 禁用训练监控

## 🔄 断点恢复机制

### 自动检测功能
脚本会自动检测现有的训练记录：
```
📁 发现训练目录: runs/small_target_train/exp
📊 检查点文件: runs/small_target_train/exp/weights/last.pt
📈 训练进度: 已完成 50 轮，下一轮: 51
📅 上次训练: 2024-05-28 14:30:15
⏱️  上次计划时长: 2小时
```

### 恢复选择
如果检测到训练记录，用户可以选择：
1. **从断点继续训练** (推荐) - 无缝继续之前的训练
2. **重新开始训练** - 删除之前的记录，从头开始
3. **取消训练** - 退出脚本

### 检查点类型
- `last.pt`: 最新的训练检查点（推荐）
- `epoch*.pt`: 特定轮次的检查点
- `best.pt`: 最佳性能检查点

## ⏱️ 定时训练功能

### 时间格式支持
```bash
# 小时格式
--time-limit 2h        # 2小时
--time-limit 1.5h      # 1.5小时

# 分钟格式
--time-limit 90m       # 90分钟
--time-limit 30m       # 30分钟

# 时:分格式
--time-limit 2:30      # 2小时30分钟
--time-limit 1:45      # 1小时45分钟

# 纯数字（默认分钟）
--time-limit 120       # 120分钟
```

### 定时控制特性
- 📊 **进度显示**: 每5分钟显示训练进度和剩余时间
- 🔄 **安全停止**: 时间到达时发送SIGINT信号，保证检查点完整保存
- 💾 **会话信息**: 自动保存训练会话信息，支持后续继续训练

### 定时训练示例
```bash
# 启动2小时定时训练
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 2h \
    --auto-resume

# 训练过程中的输出
⏱️  定时训练已启动，计划运行时间: 2小时
🕐 开始时间: 2024-05-28 14:30:15
⏳ 训练进度: 25.0% | 剩余时间: 1小时30分
⏰ 定时训练时间到达(2小时)，正在安全停止训练...
📤 已发送停止信号...
✅ 训练已安全停止
🔄 训练已按计划时间停止，检查点已保存，可稍后继续训练
```

## 📊 监控集成

### 自动监控启动
脚本会自动启动训练监控：
- 延迟30秒启动，等待训练稳定
- 自动检测结果文件存在
- 提供实时图表显示

### 监控功能
- 📈 实时损失曲线
- 📊 性能指标追踪
- 🔍 异常检测和报警
- 💾 训练日志分析

## 📂 输出文件说明

### 训练输出目录结构
```
runs/small_target_train/exp/
├── weights/
│   ├── best.pt              # 最佳模型权重
│   ├── last.pt              # 最新检查点
│   └── epoch*.pt            # 定期保存的检查点
├── results.csv              # 训练结果数据
├── hyp.yaml                 # 使用的超参数
├── opt.yaml                 # 训练选项
├── training_session.json   # 训练会话信息
└── events.out.tfevents.*   # TensorBoard日志
```

### 会话信息文件
`training_session.json` 包含：
```json
{
  "start_time": "2024-05-28T14:30:15",
  "planned_duration_seconds": 7200,
  "config": { ... },
  "session_id": "session_1716887415"
}
```

## ⚠️ 常见问题和解决方案

### 1. 训练中断恢复
```bash
# 问题：训练意外中断，如何恢复？
# 解决方案：使用自动恢复功能
python scripts/start_training.py \
    --data data/dataset.yaml \
    --auto-resume
```

### 2. 时间格式错误
```bash
# 问题：时间格式不正确
# 错误示例：
--time-limit 2小时        # ❌ 不支持中文

# 正确示例：
--time-limit 2h          # ✅ 正确格式
--time-limit 2:00        # ✅ 正确格式
--time-limit 120         # ✅ 正确格式（分钟）
```

### 3. 检查点文件损坏
```bash
# 问题：检查点文件损坏无法加载
# 解决方案：强制重新开始训练
python scripts/start_training.py \
    --data data/dataset.yaml \
    --force-restart
```

### 4. 监控无法启动
```bash
# 问题：监控功能无法正常启动
# 解决方案：禁用监控或手动启动
python scripts/start_training.py \
    --data data/dataset.yaml \
    --no-monitor

# 或手动启动监控
python scripts/training_monitor.py --results runs/train/exp/results.csv
```

### 5. 显存不足
```bash
# 问题：批次大小过大导致显存不足
# 解决方案：减小批次大小
python scripts/start_training.py \
    --data data/dataset.yaml \
    --batch-size 8           # 减小批次大小
    --imgsz 416              # 或减小图像尺寸
```

## 🔧 高级用法

### 1. 多阶段训练
```bash
# 阶段1：快速训练（小尺寸）
python scripts/start_training.py \
    --data data/dataset.yaml \
    --imgsz 416 \
    --epochs 100 \
    --name stage1 \
    --time-limit 1h

# 阶段2：精细训练（大尺寸）
python scripts/start_training.py \
    --data data/dataset.yaml \
    --weights runs/small_target_train/stage1/weights/best.pt \
    --imgsz 832 \
    --epochs 100 \
    --name stage2 \
    --time-limit 2h
```

### 2. 实验对比
```bash
# 实验A：保守超参数
python scripts/start_training.py \
    --data data/dataset.yaml \
    --hyp data/hyps/hyp.small-target-conservative.yaml \
    --name conservative_exp

# 实验B：激进超参数
python scripts/start_training.py \
    --data data/dataset.yaml \
    --hyp data/hyps/hyp.small-target-aggressive.yaml \
    --name aggressive_exp
```

### 3. 批量实验
```bash
# 创建批量训练脚本
for model in yolov5s yolov5m yolov5l; do
    python scripts/start_training.py \
        --data data/dataset.yaml \
        --cfg models/${model}.yaml \
        --weights weights/${model}.pt \
        --name ${model}_experiment \
        --time-limit 3h \
        --auto-resume
done
```

## 💡 最佳实践

### 1. 训练前准备
- 确保数据集路径正确
- 检查GPU显存是否充足
- 备份重要的配置文件
- 设置合理的训练时间限制

### 2. 训练过程中
- 定期检查训练进度
- 监控显存使用情况
- 观察损失收敛情况
- 及时调整学习率或批次大小

### 3. 训练后处理
- 评估最佳模型性能
- 生成训练报告
- 备份重要的检查点
- 清理临时文件

## 🔗 相关文档

- [small_target_train.md](small_target_train.md) - 核心训练脚本
- [training_monitor.md](training_monitor.md) - 训练监控工具
- [evaluate_detection.md](evaluate_detection.md) - 模型评估
- [gui_test.md](gui_test.md) - 图形界面测试