# small_target_train.py - 小目标检测训练脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/small_target_train.py`

## 📋 功能概述

这是小目标检测模型的专用训练脚本，基于YOLOv5架构优化，专门针对小目标检测场景进行了参数调优和功能增强。

### 主要特性
- 🎯 小目标检测专用优化
- 📊 实时训练监控和日志
- 🔧 灵活的超参数配置
- 💾 自动模型保存和恢复
- 📈 集成可视化和评估
- ⚠️ NaN检测与自动恢复机制

## 🚀 使用方法

### 基础使用
```bash
# 使用默认配置训练
python scripts/small_target_train.py --data data/dataset.yaml --epochs 200

# 指定权重和设备
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --weights yolov5s.pt \
    --epochs 200 \
    --device 0
```

### 高级使用
```bash
# 小目标优化训练
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --epochs 300 \
    --batch-size 32 \
    --imgsz 832 \
    --hyp data/hyps/hyp.yaml \
    --device 0,1 \
    --name small_target_v1

# 恢复训练
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --weights runs/small_target_train/exp/weights/last.pt \
    --resume

# 启用训练稳定性修复
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --hyp data/hyps/hyp.yaml \
    --fix-instability \
    --name stable_training
```

## 📝 参数详解

### 必需参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--data` | str | 无 | 数据集配置文件路径 | 命令行 |

### 模型配置参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--cfg` | str | `models/yolov5s.yaml` | 模型配置文件 | 命令行 |
| `--weights` | str | `yolov5s.pt` | 预训练权重路径 | 命令行 |
| `--hyp` | str | `data/hyps/hyp.yaml` | 超参数配置文件 | 命令行 |

### 训练配置参数
| 参数 | 类型 | 范围 | 默认值 | 说明 | 修改位置 |
|------|------|------|--------|------|----------|
| `--epochs` | int | 1-1000 | 300 | 训练轮次 | 命令行 |
| `--batch-size` | int | 1-128 | 16 | 批次大小 | 命令行 |
| `--imgsz` | int | 320-1280 | 640 | 输入图像尺寸 | 命令行 |
| `--device` | str | cpu,0,1,0,1 | '' | 训练设备 | 命令行 |
| `--workers` | int | 0-16 | 8 | 数据加载线程数 | 命令行 |
| `--fix-instability` | bool | False | 应用训练稳定性修复 | 命令行 |

### 优化参数
| 参数 | 类型 | 范围 | 默认值 | 说明 | 修改位置 |
|------|------|------|--------|------|----------|
| `--lr0` | float | 0.0001-0.1 | 0.0005 | 初始学习率(已调低) | 命令行/hyp文件 |
| `--lrf` | float | 0.01-1.0 | 0.005 | 最终学习率比例(已调低) | 命令行/hyp文件 |
| `--momentum` | float | 0.8-0.99 | 0.937 | SGD动量 | 命令行/hyp文件 |
| `--weight-decay` | float | 0.0-0.001 | 0.0005 | 权重衰减 | 命令行/hyp文件 |
| `--warmup-epochs` | float | 0.0-10.0 | 10.0 | 预热轮次(已增加) | hyp文件 |
| `--gradient-clip-norm` | float | 1.0-10.0 | 1.0 | 梯度裁剪强度(已增强) | hyp文件 |

### 数据增强参数
| 参数 | 类型 | 范围 | 默认值 | 说明 | 修改位置 |
|------|------|------|--------|------|----------|
| `--hsv-h` | float | 0.0-1.0 | 0.015 | HSV色调增强 | hyp文件 |
| `--hsv-s` | float | 0.0-1.0 | 0.7 | HSV饱和度增强 | hyp文件 |
| `--hsv-v` | float | 0.0-1.0 | 0.4 | HSV明度增强 | hyp文件 |
| `--degrees` | float | 0.0-45.0 | 0.0 | 旋转角度范围 | hyp文件 |
| `--translate` | float | 0.0-0.5 | 0.1 | 平移范围 | hyp文件 |
| `--scale` | float | 0.0-1.0 | 0.5 | 缩放范围 | hyp文件 |
| `--shear` | float | 0.0-10.0 | 0.0 | 剪切变换 | hyp文件 |
| `--perspective` | float | 0.0-0.001 | 0.0 | 透视变换 | hyp文件 |
| `--flipud` | float | 0.0-1.0 | 0.0 | 上下翻转概率 | hyp文件 |
| `--fliplr` | float | 0.0-1.0 | 0.5 | 左右翻转概率 | hyp文件 |
| `--mosaic` | float | 0.0-1.0 | 1.0 | Mosaic增强概率 | hyp文件 |
| `--mixup` | float | 0.0-1.0 | 0.0 | MixUp增强概率 | hyp文件 |
| `--copy_paste` | float | 0.0-1.0 | 0.0 | Copy-Paste增强概率 | hyp文件 |

### 损失函数参数
| 参数 | 类型 | 范围 | 默认值 | 说明 | 修改位置 |
|------|------|------|--------|------|----------|
| `--box` | float | 0.01-10.0 | 0.05 | 边界框损失权重 | hyp文件 |
| `--cls` | float | 0.01-10.0 | 0.3 | 分类损失权重 | hyp文件 |
| `--obj` | float | 0.01-10.0 | 0.4 | 目标性损失权重(已平衡) | hyp文件 |
| `--fl_gamma` | float | 0.0-5.0 | 0.0 | Focal Loss gamma参数 | hyp文件 |

### 输出控制参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--project` | str | `runs/small_target_train` | 项目目录 | 命令行 |
| `--name` | str | `exp` | 实验名称 | 命令行 |
| `--save-period` | int | -1 | 权重保存间隔(轮次) | 命令行 |
| `--resume` | bool | False | 恢复训练 | 命令行 |
| `--exist-ok` | bool | False | 覆盖现有实验 | 命令行 |

## 📂 输入文件格式

### 1. 数据集配置文件 (dataset.yaml)
```yaml
# 数据集路径
path: /path/to/dataset  # 数据集根目录
train: images/train     # 训练图片相对路径
val: images/val         # 验证图片相对路径
test: images/test       # 测试图片相对路径(可选)

# 类别数量和名称
nc: 1                   # 类别数量
names: ['target']       # 类别名称列表

# 下载设置(可选)
download: |
  # 下载脚本
```

### 2. 超参数配置文件 (hyp.yaml)
```yaml
# 学习率设置
lr0: 0.0005             # 初始学习率(降低为更稳定值)
lrf: 0.005              # 最终学习率比例(降低为更平缓衰减)
momentum: 0.937         # SGD动量
weight_decay: 0.0005    # 权重衰减
warmup_epochs: 10.0     # 预热轮次(增加为更稳定训练)
warmup_momentum: 0.8    # 预热动量
warmup_bias_lr: 0.02    # 预热偏置学习率(降低为更稳定)

# 损失函数权重
box: 0.05               # 边界框损失权重
cls: 0.3                # 分类损失权重
cls_pw: 1.0             # 分类正样本权重
obj: 0.4                # 目标性损失权重(降低为更平衡训练)
obj_pw: 1.0             # 目标性正样本权重
iou_t: 0.20             # IoU训练阈值
anchor_t: 4.0           # 锚框倍数阈值
fl_gamma: 0.0           # Focal Loss gamma
gradient_clip_norm: 1.0 # 梯度裁剪强度(增强为更稳定训练)

# 数据增强参数
hsv_h: 0.015            # 色调增强
hsv_s: 0.7              # 饱和度增强
hsv_v: 0.4              # 明度增强
degrees: 0.0            # 旋转角度
translate: 0.1          # 平移范围
scale: 0.5              # 缩放范围
shear: 0.0              # 剪切变换
perspective: 0.0        # 透视变换
flipud: 0.0             # 上下翻转
fliplr: 0.5             # 左右翻转
mosaic: 1.0             # Mosaic增强
mixup: 0.0              # MixUp增强
copy_paste: 0.0         # Copy-Paste增强
```

## 📊 输出文件说明

### 训练输出目录结构
```
runs/small_target_train/exp/
├── weights/                    # 模型权重文件
│   ├── best.pt                 # 最佳模型权重
│   ├── last.pt                 # 最新模型权重
│   └── epoch_*.pt              # 特定轮次权重(如果设置保存间隔)
├── results.csv                 # 训练结果数据
├── results.png                 # 训练曲线图
├── confusion_matrix.png        # 混淆矩阵
├── labels_correlogram.jpg      # 标签相关性图
├── labels.jpg                  # 标签分布图
├── train_batch*.jpg            # 训练样本批次图
├── val_batch*_labels.jpg       # 验证标签批次图
├── val_batch*_pred.jpg         # 验证预测批次图
├── F1_curve.png               # F1曲线
├── P_curve.png                # 精确度曲线
├── R_curve.png                # 召回率曲线
├── PR_curve.png               # PR曲线
├── hyp.yaml                   # 使用的超参数
├── opt.yaml                   # 训练选项配置
└── events.out.tfevents.*      # TensorBoard日志
```

### 主要输出文件解释

#### 1. 模型权重文件
- **best.pt**: 验证集上性能最佳的模型权重
- **last.pt**: 训练结束时的最新权重，用于恢复训练
- **epoch_N.pt**: 特定轮次的权重，便于回滚

#### 2. 训练结果数据 (results.csv)
```csv
epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,val/box_loss,val/obj_loss,val/cls_loss,lr/pg0,lr/pg1,lr/pg2
0,0.05234,0.02145,0.00156,0.0,0.0,0.0,0.0,0.04987,0.02089,0.00134,0.01,0.01,0.01
```

#### 3. 可视化图表
- **results.png**: 综合训练曲线(损失、指标、学习率)
- **confusion_matrix.png**: 验证集混淆矩阵
- **F1_curve.png**: F1分数vs置信度曲线
- **PR_curve.png**: 精确度-召回率曲线

#### 4. 数据分析图表
- **labels.jpg**: 数据集标签分布统计
- **labels_correlogram.jpg**: 标签尺寸相关性分析
- **train_batch*.jpg**: 训练批次样本可视化
- **val_batch*_labels.jpg**: 验证集真实标签
- **val_batch*_pred.jpg**: 验证集预测结果

## 🎯 小目标优化策略

### 1. 模型架构优化
```yaml
# 推荐的小目标模型配置
anchors:
  # 针对小目标优化的锚框尺寸
  - [4,5, 8,10, 13,16]          # 更小的P3锚框
  - [23,29, 43,55, 73,105]      # 中等P4锚框  
  - [146,217, 231,300, 335,433] # 大型P5锚框
```

### 2. 超参数调优
```yaml
# 小目标专用超参数
lr0: 0.0005           # 降低学习率提高稳定性
box: 0.05             # 基础边界框损失
obj: 0.4              # 降低目标性损失平衡训练
mosaic: 0.8           # 适度Mosaic增强
copy_paste: 0.0       # 不使用Copy-Paste增强
scale: 0.5            # 中等缩放范围
gradient_clip_norm: 1.0 # 强梯度裁剪防止爆炸
warmup_epochs: 10.0   # 延长预热期提高稳定性
```

### 3. 训练稳定性增强
```bash
# 稳定训练命令
python scripts/small_target_train.py \
    --data data/small_target.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --epochs 300 \
    --batch-size 16 \             # 避免过大批次
    --imgsz 640 \
    --hyp data/hyps/hyp.yaml \    # 使用优化超参数
    --device 0 \
    --fix-instability \           # 启用稳定性修复
    --name small_target_stable
```

### 4. NaN检测与自动恢复
脚本内置了自动NaN检测和恢复机制，当检测到以下情况时会自动处理：
- 连续3批次出现NaN损失值时自动降低学习率
- 自动跳过NaN损失批次，避免训练崩溃
- 训练过程中记录NaN警告并显示在日志中

## ⚠️ 常见问题和解决方案

### 1. 显存不足
```bash
# 问题：CUDA out of memory
# 解决方案：
# 1. 减少批次大小
python scripts/small_target_train.py --batch-size 8

# 2. 减少图像尺寸
python scripts/small_target_train.py --imgsz 640

# 3. 使用梯度累积
python scripts/small_target_train.py --batch-size 8 --accumulate 4
```

### 2. 数据集路径错误
```bash
# 问题：FileNotFoundError或数据集找不到
# 解决方案：检查dataset.yaml配置
cat data/dataset.yaml
# 确保路径正确且图片和标签文件存在
```

### 3. 训练不收敛
```bash
# 问题：损失不下降或指标不提升
# 解决方案：
# 1. 使用稳定性修复参数
python scripts/small_target_train.py --fix-instability

# 2. 手动降低学习率
python scripts/small_target_train.py --lr0 0.0002

# 3. 增加预热轮次
# 修改hyp.yaml中的warmup_epochs: 15.0

# 4. 检查数据质量和标注
python scripts/check_dataset.py --data data/dataset.yaml
```

### 4. 训练中断恢复
```bash
# 问题：训练意外中断
# 解决方案：使用last.pt恢复训练
python scripts/small_target_train.py \
    --weights runs/small_target_train/exp/weights/last.pt \
    --resume
```

### 5. NaN 损失问题
```bash
# 问题：出现NaN损失导致训练不稳定
# 解决方案：
# 1. 使用稳定训练配置
python scripts/small_target_train.py --fix-instability

# 2. 手动强化梯度裁剪
# 修改hyp.yaml中gradient_clip_norm: 0.5

# 3. 使用保守学习率
python scripts/small_target_train.py --lr0 0.0001

# 4. 确保数据标签正确格式化
# 检查标签文件是否有异常值或不合规范的数据点
```

## 🔧 高级训练技巧

### 1. 多GPU训练
```bash
# 使用多GPU加速训练
python -m torch.distributed.launch --nproc_per_node 2 \
    scripts/small_target_train.py \
    --data data/dataset.yaml \
    --epochs 200 \
    --batch-size 64 \
    --device 0,1
```

### 2. 混合精度训练
```bash
# 启用自动混合精度(AMP)提升速度
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --epochs 200 \
    --amp
```

### 3. 渐进式训练
```bash
# 先用小尺寸训练，再用大尺寸精调
# 阶段1: 小尺寸快速收敛
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --imgsz 416 \
    --epochs 100 \
    --name stage1

# 阶段2: 大尺寸精调
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --weights runs/small_target_train/stage1/weights/best.pt \
    --imgsz 832 \
    --epochs 100 \
    --lr0 0.001 \
    --name stage2
```

### 4. 超参数自动调优
```bash
# 使用进化算法自动调优超参数
python scripts/small_target_train.py \
    --data data/dataset.yaml \
    --epochs 200 \
    --evolve 300
```

## 📈 训练监控和分析

### 1. TensorBoard可视化
```bash
# 启动TensorBoard监控训练
tensorboard --logdir runs/small_target_train
# 在浏览器中访问 http://localhost:6006
```

### 2. 实时监控指标
- **损失曲线**: 监控train/val loss收敛情况
- **性能指标**: 观察mAP、precision、recall变化
- **学习率**: 确认学习率调度是否合理
- **数据可视化**: 检查数据增强效果

### 3. 训练后分析
```bash
# 生成详细的训练图表
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv

# 评估训练好的模型
python scripts/evaluate_detection.py \
    --weights runs/small_target_train/exp/weights/best.pt
```

## 🔗 相关文档

- [evaluate_detection.md](evaluate_detection.md) - 模型评估分析
- [generate_charts.md](generate_charts.md) - 训练图表生成
- [gui_test.md](gui_test.md) - 图形界面测试
- [training_monitor.md](training_monitor.md) - 实时训练监控