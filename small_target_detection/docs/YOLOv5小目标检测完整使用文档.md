# YOLOv5 小目标检测完整使用文档

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日

## 目录
1. [脚本概述与环境配置](#1-脚本概述与环境配置)
2. [依赖包详细说明](#2-依赖包详细说明)
3. [训练脚本使用指南](#3-训练脚本使用指南)
4. [断点继续训练功能](#4-断点继续训练功能)
5. [定时训练功能](#5-定时训练功能)
6. [最佳实践与故障排除](#6-最佳实践与故障排除)
7. [模型评估功能（evaluate_detection.py）](#7-模型评估功能evaluate_detectionpy)
8. [评估结果对比功能（compare_evaluation_results.py）](#8-评估结果对比功能compare_evaluation_resultspy)
9. [训练图表生成功能（generate_charts.py）](#9-训练图表生成功能generate_chartspy)
10. [训练监控功能（training_monitor.py）](#10-训练监控功能training_monitorpy)
11. [图形界面测试工具（gui_test.py）](#11-图形界面测试工具gui_testpy)
12. [标签类别修改工具（update_labels_class.py）](#12-标签类别修改工具update_labels_classpy)
13. [视频检测工具（test_videos.py）](#13-视频检测工具test_videospy)

---

## 1. 脚本概述与环境配置

### 1.1 脚本概述

`small_target_train.py` 是一个专门针对小目标检测任务的深度学习训练脚本。它基于YOLOv5算法，专门优化了对小物体（如田径跑道标记）的检测能力。

### 什么是深度学习训练？
- **训练**：让计算机通过大量图片学习如何识别特定物体
- **小目标**：图片中很小的物体，比如远处的汽车、小标记等
- **YOLOv5**：一种先进的目标检测算法，能同时识别多个物体并标出位置

### 1.2 环境安装

#### Python环境要求
- Python 3.7 或更高版本
- CUDA 11.0+ (如果使用GPU)

#### 安装依赖包
```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果出现版本冲突，可以指定版本安装
pip install albumentations==1.3.0
pip install torch>=1.7.0 torchvision>=0.8.1
```

#### 验证安装
```python
import torch
import albumentations as A
import cv2
print(f"PyTorch版本: {torch.__version__}")
print(f"Albumentations版本: {A.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

---

## 2. 依赖包详细说明

### 2.1 核心框架依赖

#### PyTorch生态系统
- **torch**: PyTorch深度学习框架的核心包
  - 提供张量计算、自动求导、神经网络模块
  - 支持GPU加速计算
  - YOLOv5的基础计算引擎

- **torchvision**: PyTorch官方视觉库
  - 提供常用的图像变换函数
  - 包含预训练的计算机视觉模型
  - 支持常见数据集的加载

### 2.2 数据处理和增强

#### 图像处理核心库
- **opencv-python**: 计算机视觉处理库
  - 图像读取、保存、格式转换
  - 基础图像变换（缩放、旋转、裁剪）
  - 绘制检测框和标注信息
  - 视频处理和摄像头接口

- **Pillow**: Python图像库
  - 支持多种图像格式（JPEG、PNG、GIF等）
  - 基础图像操作和滤镜效果
  - 与numpy数组的便捷转换

- **albumentations**: 专业数据增强库
  - **核心优势**：
    - 处理速度比传统方法快10-40倍
    - 原生支持目标检测边界框变换
    - 提供100+种增强方法
    - 支持多种数据格式（图像、掩码、边界框、关键点）
  - **增强方法分类**：
    - 几何变换：旋转、缩放、翻转、透视变换
    - 颜色调整：亮度、对比度、饱和度、色调
    - 噪声添加：高斯噪声、椒盐噪声、ISO噪声
    - 模糊效果：高斯模糊、运动模糊、中值模糊
    - 天气模拟：雾、雨、雪、阴影、眩光
    - 压缩模拟：JPEG压缩、图像质量降级

### 2.3 数值计算和科学计算

#### 基础数学库
- **numpy**: 数值计算基础库
  - 多维数组操作
  - 线性代数运算
  - 随机数生成
  - 图像数据的数组表示

- **scipy**: 高级科学计算库
  - 优化算法（损失函数优化）
  - 信号处理（图像滤波）
  - 统计分析（数据分布分析）
  - 稀疏矩阵操作

### 2.4 可视化和监控

#### 训练监控
- **tensorboard**: 训练可视化工具
  - 实时监控损失曲线
  - 可视化网络结构
  - 显示图像和检测结果
  - 超参数对比分析

- **matplotlib**: 基础绘图库
  - 生成训练曲线图
  - 绘制混淆矩阵
  - 可视化数据分布
  - 创建检测结果图表

- **seaborn**: 统计可视化库
  - 美化图表样式
  - 高级统计图表
  - 数据分布可视化
  - 相关性分析图

#### 数据分析
- **pandas**: 数据处理和分析
  - 训练日志数据整理
  - 实验结果统计
  - 数据表格操作
  - CSV文件读写

### 2.5 系统工具和辅助

#### 系统监控
- **psutil**: 系统资源监控
  - 实时监控CPU使用率
  - 内存占用情况
  - GPU状态查询
  - 进程管理

- **tqdm**: 进度条显示
  - 训练进度可视化
  - 数据加载进度
  - 批处理进度跟踪
  - 估算剩余时间

#### 文件和配置
- **PyYAML**: YAML文件处理
  - 读取数据集配置文件
  - 解析超参数文件
  - 保存实验配置
  - 模型参数序列化

- **requests**: HTTP请求处理
  - 下载预训练模型
  - 获取在线数据集
  - API接口调用
  - 文件传输

### 2.6 性能分析和优化

#### 模型分析
- **thop**: 模型复杂度计算
  - 计算FLOPs（浮点运算次数）
  - 评估模型计算复杂度
  - 参数量统计
  - 推理速度评估

---

## 3. 训练脚本使用指南

### 3.1 脚本主要组成部分

#### SmallTargetAugmentation类（数据增强器）
```python
class SmallTargetAugmentation:
    """小目标检测专用数据增强类"""
```

**作用**：让训练数据更丰富多样
**通俗解释**：就像给同一张照片加不同的滤镜、旋转、调亮度等，让AI见过更多变化的样子

**包含的变化类型**：
- **几何变换**：旋转、缩放、翻转、平移
- **颜色调整**：亮度、对比度、饱和度、色调
- **噪声模拟**：高斯噪声、ISO噪声、乘性噪声
- **模糊效果**：高斯模糊、运动模糊、中值模糊
- **遮挡模拟**：随机擦除、方块遮挡
- **天气效果**：雾、阴影、眩光

### 3.2 命令行参数详细说明

#### 基础设置参数

##### --weights (初始权重路径)
```bash
--weights weights/yolov5s.pt
```
- **含义**：预训练模型的起点
- **通俗解释**：就像找一个已经学过基础知识的学生，再教他专门技能
- **推荐选择**：
  - `yolov5s.pt`：小模型，速度快（推荐小目标检测）
  - `yolov5m.pt`：中等模型，平衡性能
  - `yolov5l.pt`：大模型，精度高但慢

##### --cfg (模型配置文件)
```bash
--cfg models/yolov5s.yaml
```
- **含义**：定义神经网络的结构
- **通俗解释**：就像建筑图纸，规定大脑有多少层、每层多少个神经元

##### --data (数据集配置文件)
```bash
--data data/track_markers.yaml
```
- **含义**：告诉程序训练图片在哪里
- **必须包含的内容**：
```yaml
path: ./dataset  # 数据集根目录
train: train/images  # 训练图片路径
val: val/images      # 验证图片路径
nc: 1                # 类别数量
names: ['marker']    # 类别名称
```

#### 训练控制参数

##### --epochs (训练轮次)
```bash
--epochs 200
```
- **含义**：整个数据集学习多少遍
- **推荐设置**：
  - 小数据集（<500张）：100-200轮
  - 中等数据集（500-2000张）：150-300轮
  - 大数据集（>2000张）：100-200轮

##### --batch-size (批次大小)
```bash
--batch-size 16
```
- **含义**：每次同时处理多少张图片
- **根据显存调整**：
  - 4GB显存：batch-size 8
  - 8GB显存：batch-size 16-32
  - 12GB+ 显存：batch-size 32-64

##### --imgsz (图像尺寸)
```bash
--imgsz 640
```
- **含义**：输入图片的大小（像素）
- **小目标推荐**：
  - 标准：640像素
  - 高精度：832或1024像素
  - 快速训练：512像素

### 3.3 完整使用示例

#### 基础训练命令（推荐新手）
```bash
python small_target_train.py \
    --data data/track_markers.yaml \
    --hyp data/hyps/hyp.small-target-conservative.yaml \
    --epochs 150 \
    --batch-size 16 \
    --imgsz 640 \
    --weights weights/yolov5s.pt \
    --project runs/track_detection \
    --name beginner_exp \
    --device 0 \
    --augment-ratio 0.3
```

#### 标准训练命令（推荐使用）
```bash
python small_target_train.py \
    --data data/track_markers.yaml \
    --hyp data/hyps/hyp.small-target.yaml \
    --epochs 200 \
    --batch-size 16 \
    --imgsz 640 \
    --weights weights/yolov5s.pt \
    --project runs/track_detection \
    --name standard_exp \
    --device 0 \
    --multi-scale \
    --cos-lr \
    --augment-ratio 0.35 \
    --augment-schedule progressive \
    --cache ram \
    --workers 8
```

---

## 4. 断点继续训练功能（start_training.py）

### 4.1 功能概述

`start_training.py` 是一个功能强大的训练启动器，提供智能断点恢复、定时训练、实时监控等高级功能，让训练过程更加便捷和可控。

#### 主要特性
- 🔄 **智能断点恢复**：自动检测和恢复训练进度
- ⏱️ **定时训练控制**：支持多种时间格式的训练时间限制
- 📊 **实时训练监控**：集成训练监控功能
- 🎯 **小目标检测专用优化**：针对小目标检测任务优化
- 💾 **训练会话管理**：记录和管理训练会话信息
- 🔧 **灵活的参数配置**：支持各种训练参数配置

### 4.2 断点恢复机制详解

#### 自动检测功能
脚本会智能检测现有的训练记录：
```
📁 发现训练目录: runs/small_target_train/exp
📊 检查点文件: runs/small_target_train/exp/weights/last.pt
📈 训练进度: 已完成 50 轮，下一轮: 51
📅 上次训练: 2024-05-28 14:30:15
⏱️  上次计划时长: 2小时
```

#### 检查点文件优先级
1. **last.pt** - 最新的训练检查点（优先使用）
2. **epoch*.pt** - 特定轮次的检查点
3. **best.pt** - 最佳性能检查点

#### 恢复方式选择
如果检测到训练记录，提供三种选择：
1. **从断点继续训练**（推荐）- 无缝继续之前的训练
2. **重新开始训练** - 删除之前的记录，从头开始
3. **取消训练** - 退出脚本

### 4.3 使用方法

#### 交互式断点恢复（默认模式）
```bash
python scripts/start_training.py --data data/dataset.yaml --name my_experiment
```
如果检测到检查点，会显示选择菜单：
```
🔄 检测到已有训练记录，请选择:
1. 从断点继续训练 (推荐)
2. 重新开始训练
3. 取消训练

请输入选择 (1/2/3):
```

#### 自动断点恢复
```bash
# 自动从断点继续训练，不询问用户
python scripts/start_training.py \
    --data data/dataset.yaml \
    --name my_experiment \
    --auto-resume
```

#### 强制重新开始
```bash
# 强制重新开始训练，忽略所有检查点
python scripts/start_training.py \
    --data data/dataset.yaml \
    --name my_experiment \
    --force-restart
```

### 4.4 断点恢复参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--auto-resume` | flag | False | 自动从断点继续训练，不询问用户 |
| `--force-restart` | flag | False | 强制重新开始训练，忽略所有检查点 |

### 4.5 工作原理

#### 检查点检测流程
```
开始训练启动器
    ↓
检查训练目录是否存在
    ↓
查找检查点文件 (last.pt 或 epoch*.pt)
    ↓
读取训练状态 (results.csv)
    ↓
根据参数决定继续方式 (交互/自动/强制)
    ↓
启动训练或恢复训练
```

#### 检查点文件结构
```
runs/small_target_train/exp/
├── weights/
│   ├── last.pt          # 最新检查点（优先使用）
│   ├── best.pt          # 最佳模型
│   └── epoch*.pt        # 定期保存的检查点
├── results.csv          # 训练结果记录
├── training_session.json # 训练会话信息（新增）
├── hyp.yaml            # 超参数配置
└── opt.yaml            # 训练选项配置
```

---

## 5. 定时训练功能

### 5.1 功能概述

定时训练功能允许设置训练时间限制，到达指定时间后自动安全停止训练并保存检查点，便于利用空闲时间逐步完成长期训练任务。

### 5.2 时间格式支持

#### 支持的时间格式
- **小时格式**: `2h`, `1.5h`, `0.5h`
- **分钟格式**: `30m`, `90m`, `120m`
- **秒格式**: `3600s`, `1800s`
- **时:分格式**: `2:30`, `1:45`, `0:30`
- **纯数字**: `120` (默认为分钟)

#### 时间格式示例
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

### 5.3 定时训练特性

#### 智能训练控制
- **安全停止机制**：发送SIGINT信号优雅停止训练
- **自动保存检查点**：确保训练状态完整保存
- **实时进度显示**：每5分钟显示训练进度和剩余时间
- **会话信息记录**：自动记录训练会话信息

#### 进度监控显示
训练过程中会显示：
```
⏱️  定时训练已启动，计划运行时间: 2小时
🕐 开始时间: 2024-05-28 14:30:15
⏳ 训练进度: 25.0% | 剩余时间: 1小时30分
⏰ 定时训练时间到达(2小时)，正在安全停止训练...
📤 已发送停止信号...
✅ 训练已安全停止
⏱️  实际训练时间: 1小时59分45秒
🔄 训练已按计划时间停止，检查点已保存，可稍后继续训练
```

### 5.4 使用方法

#### 基本定时训练
```bash
# 训练2小时
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 2h \
    --name exp_2h

# 训练30分钟
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 30m \
    --name exp_30m

# 训练1小时30分钟
python scripts/start_training.py \
    --data data/dataset.yaml \
    --time-limit 1:30 \
    --name exp_1h30m
```

#### 结合断点继续训练
```bash
# 第一次训练2小时
python scripts/start_training.py \
    --epochs 500 \
    --time-limit 2h \
    --name long_training

# 第二次继续训练2小时（自动从断点继续）
python scripts/start_training.py \
    --epochs 500 \
    --time-limit 2h \
    --name long_training \
    --auto-resume

# 第三次继续训练2小时
python scripts/start_training.py \
    --epochs 500 \
    --time-limit 2h \
    --name long_training \
    --auto-resume
```

#### 完整配置定时训练
```bash
python scripts/start_training.py \
    --data data/small_target.yaml \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --epochs 500 \
    --batch-size 32 \
    --imgsz 832 \
    --device 0 \
    --time-limit 4h \
    --auto-resume \
    --save-period 5 \
    --name production_training
```

### 5.5 定时训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--time-limit` | str | None | 训练时间限制 (格式: 2h, 30m, 2:30, 120) |
| `--schedule-training` | flag | False | 启用定时训练模式 (显示更多时间信息) |

### 5.6 定时训练参数一览表

| 参数 | 原始值 | 优化值 | 影响 | 建议场景 |
|----------|------|--------|----------|----------|
| **lr0** | 0.002 | 0.0005 | 学习率降低4倍，避免梯度爆炸 | 所有训练场景 |
| **lrf** | 0.01 | 0.005 | 更平缓的学习率衰减 | 长时间训练 |
| **warmup_epochs** | 5.0 | 10.0 | 延长预热期，提高初期稳定性 | 复杂数据集 |
| **warmup_bias_lr** | 0.05 | 0.02 | 降低预热期学习率 | 高难度目标 |
| **obj** | 0.7 | 0.4 | 降低目标性损失权重 | 小目标检测 |
| **gradient_clip_norm** | 5.0 | 1.0 | 强化梯度裁剪强度 | 所有训练场景 |

### 5.7 定时控制流程

#### 工作原理
```
启动训练脚本
    ↓
解析时间限制参数
    ↓
启动训练进程
    ↓
并行启动定时器和进度监控
    ↓
到达时间限制 → 发送SIGINT信号
    ↓
训练进程安全停止并保存检查点
    ↓
记录会话信息和训练统计
```

#### 会话信息管理
`training_session.json` 文件记录：
```json
{
  "start_time": "2024-05-28T14:30:15",
  "planned_duration_seconds": 7200,
  "actual_duration_seconds": 7185,
  "config": {
    "data": "data/dataset.yaml",
    "epochs": 200,
    "batch_size": 16
  },
  "session_id": "session_1716887415",
  "status": "completed_on_time"
}
```

### 5.8 应用场景

#### 场景1: 利用空闲时间训练
```bash
# 工作日中午训练1小时
python scripts/start_training.py \
    --time-limit 1h \
    --name lunch_training \
    --auto-resume

# 晚上训练3小时
python scripts/start_training.py \
    --time-limit 3h \
    --name lunch_training \
    --auto-resume

# 周末训练8小时
python scripts/start_training.py \
    --time-limit 8h \
    --name lunch_training \
    --auto-resume
```

#### 场景2: 分段式长期训练
```bash
# 每天晚上训练2小时，连续一周完成长期训练
for i in {1..7}; do
    echo "第${i}天训练"
    python scripts/start_training.py \
        --epochs 500 \
        --time-limit 2h \
        --name week_training \
        --auto-resume
done
```

### 5.9 故障排除

#### 常见问题

##### 时间格式错误
```bash
# 错误示例
python start_training.py --time-limit "2小时"  # ❌
python start_training.py --time-limit "2:30:45"  # ❌

# 正确示例
python start_training.py --time-limit "2h"     # ✅
python start_training.py --time-limit "2:30"   # ✅
```

##### 训练无法停止
```bash
# 如果定时停止失败，手动终止
ps aux | grep start_training.py
kill -SIGINT <进程ID>
```

##### 会话信息丢失
如果 `training_session.json` 文件损坏或丢失，可以通过 `results.csv` 文件手动恢复训练进度。

---

## 6. 最佳实践与故障排除

### 6.1 数据准备建议
- **图片数量**：至少200-300张，推荐500+张
- **标注质量**：确保边界框准确，避免漏标
- **数据多样性**：包含不同光照、角度、场景
- **验证集**：保留20%数据作为验证集

### 6.2 训练策略建议
- **从预训练模型开始**：使用yolov5s.pt或yolov5m.pt
- **渐进式训练**：从保守配置开始，逐步优化
- **监控过拟合**：关注验证精度变化
- **训练稳定性优先**：使用已优化的超参数配置
- **多次实验**：尝试不同参数组合

### 6.3 常见问题和解决方案

#### 安装问题

##### albumentations安装失败
```bash
ERROR: Could not find a version that satisfies the requirement albumentations
```
**解决方案**：
```bash
# 升级pip
pip install --upgrade pip

# 尝试指定版本安装
pip install albumentations==1.3.0

# 如果还是失败，使用conda
conda install -c conda-forge albumentations
```

##### PyTorch版本冲突
**解决方案**：
```bash
# 卸载现有版本
pip uninstall torch torchvision

# 重新安装匹配版本
pip install torch==1.12.0 torchvision==0.13.0
```

#### 训练问题

##### 显存不足错误
```
CUDA out of memory
```
**解决方案**：
- 减小batch-size：`--batch-size 8`
- 减小图像尺寸：`--imgsz 512`
- 关闭缓存：不使用 `--cache`
- 减少workers：`--workers 4`

##### 训练出现NaN或无穷大损失
```
Detected NaN/Inf loss values! (连续3/3)
```
**解决方案**：
- 使用稳定性优化配置：`--fix-instability`
- 降低学习率：修改hyp.yaml中的`lr0: 0.0005`或更低
- 加强梯度裁剪：修改hyp.yaml中的`gradient_clip_norm: 0.5`
- 延长预热期：修改hyp.yaml中的`warmup_epochs: 15.0`
- 批次大小减半：`--batch-size 8`

##### 训练过慢
**解决方案**：
- 使用缓存：`--cache ram`
- 增加workers：`--workers 16`
- 检查GPU使用：`nvidia-smi`
- 降低增强比例：`--augment-ratio 0.2`

#### 定时训练问题

##### 时间格式错误
```bash
# 错误示例
python start_training.py --time-limit "2小时"  # ❌
python start_training.py --time-limit "2:30:45"  # ❌

# 正确示例
python start_training.py --time-limit "2h"     # ✅
python start_training.py --time-limit "2:30"   # ✅
```

##### 训练无法停止
```bash
# 如果定时停止失败，手动终止
ps aux | grep start_training.py
kill -SIGINT <进程ID>
```

##### 会话信息丢失
如果 `training_session.json` 文件损坏或丢失，可以通过 `results.csv` 文件手动恢复训练进度。

### 6.4 参数调优顺序
1. 先用默认参数训练，观察基础效果
2. 如果出现训练不稳定，启用`--fix-instability`
3. 调整数据增强比例（augment-ratio）
4. 尝试不同超参数配置文件
5. 调整学习率和优化器
6. 最后调整模型大小和图像尺寸

### 6.5 常用命令组合

#### 快速测试
```bash
python start_training.py --epochs 50 --batch-size 8 --imgsz 512
```

#### 标准训练
```bash
python start_training.py --hyp data/hyps/hyp.yaml --multi-scale --cos-lr
```

#### 稳定性训练
```bash
python start_training.py --fix-instability --hyp data/hyps/hyp.yaml --batch-size 8
```

#### 高精度训练
```bash
python start_training.py --hyp data/hyps/hyp.yaml --imgsz 832 --epochs 300
```

#### 定时分段训练
```bash
python start_training.py --time-limit 2h --auto-resume --save-period 5
```

### 6.6 训练稳定性参数一览表

| 参数 | 原始值 | 优化值 | 影响 | 建议场景 |
|----------|------|--------|----------|----------|
| **lr0** | 0.002 | 0.0005 | 学习率降低4倍，避免梯度爆炸 | 所有训练场景 |
| **lrf** | 0.01 | 0.005 | 更平缓的学习率衰减 | 长时间训练 |
| **warmup_epochs** | 5.0 | 10.0 | 延长预热期，提高初期稳定性 | 复杂数据集 |
| **warmup_bias_lr** | 0.05 | 0.02 | 降低预热期学习率 | 高难度目标 |
| **obj** | 0.7 | 0.4 | 降低目标性损失权重 | 小目标检测 |
| **gradient_clip_norm** | 5.0 | 1.0 | 强化梯度裁剪强度 | 所有训练场景 |

---

## 7. 模型评估功能（evaluate_detection.py）

### 7.1 功能概述

`evaluate_detection.py` 是检测结果与人工标注对比分析的专用脚本，能够计算多种检测准确性指标，支持自动实验目录生成和丰富的可视化功能。

#### 主要特性
- 🎯 **多IoU阈值评估**：支持mAP@0.5, mAP@0.5:0.95等多种评估标准
- 📊 **详细置信度分析**：分析模型的置信度分布和统计特性
- 🖼️ **自动样本图像对比**：生成预测与真实标注的可视化对比
- 📈 **丰富的可视化图表**：提供多维度的性能分析图表
- 💾 **自动实验目录管理**：智能组织和保存评估结果

### 7.2 使用方法

#### 基础评估命令
```bash
# 评估检测结果
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images

# 指定输出目录
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images \
    --output-dir runs/evaluation/custom_eval
```

#### 高级评估配置
```bash
# 自定义IoU阈值（适合小目标检测）
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images \
    --iou-thresholds 0.3 0.5 0.7 0.9 \
    --project runs/my_evaluation \
    --name small_target_eval
```

### 7.3 输入文件格式要求

#### 预测标签文件格式（YOLO + 置信度）
```
# 文件名: image_name.txt
# 格式: class_id x_center y_center width height confidence
0 0.5 0.3 0.1 0.2 0.85
0 0.7 0.6 0.15 0.25 0.92
```

#### 真实标签文件格式（标准YOLO格式）
```
# 文件名: image_name.txt  
# 格式: class_id x_center y_center width height
0 0.5 0.3 0.1 0.2
0 0.7 0.6 0.15 0.25
```

#### 文件结构示例
```
project/
├── images/                 # 图片文件
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── data/labels/           # 真实标注（Ground Truth）
│   ├── img_001.txt
│   ├── img_002.txt  
│   └── ...
└── runs/detect/exp/labels/ # 模型预测结果
    ├── img_001.txt
    ├── img_002.txt
    └── ...
```

### 7.4 评估指标详解

#### 基础检测指标
- **TP (True Positive)**: 正确检测的目标数量
- **FP (False Positive)**: 错误检测的数量（误检）
- **FN (False Negative)**: 漏检的目标数量

#### 性能评估指标
- **Precision**: 精确度 = TP/(TP+FP)，表示检测准确性
- **Recall**: 召回率 = TP/(TP+FN)，表示检测完整性  
- **F1-Score**: F1分数 = 2×(Precision×Recall)/(Precision+Recall)，综合指标
- **mAP@0.5**: IoU阈值0.5时的平均精度
- **mAP@0.5:0.95**: IoU阈值0.5到0.95的平均精度（更严格）

#### 置信度相关指标
- **平均置信度**: 所有预测的平均置信度分数
- **TP平均置信度**: 正确检测的平均置信度
- **FP平均置信度**: 误检的平均置信度
- **最佳置信度阈值**: F1分数最高时对应的置信度阈值

### 7.5 输出结果详解

#### 评估输出目录结构
```
runs/evaluation/eval/
├── config.yaml                    # 实验配置信息
├── evaluation_results.json        # 详细评估结果数据
├── overall_metrics.png            # 整体指标条形图
├── detection_distribution.png     # 检测结果分布图
├── class_ap_comparison.png        # 各类别AP对比图
├── map_vs_iou.png                # mAP vs IoU阈值曲线
├── confidence_distribution.png    # 置信度分布图
├── confidence_vs_performance.png  # 置信度阈值vs性能曲线
├── sample_images/                 # 样本图像对比
│   ├── sample_01_img_001.jpg
│   ├── sample_02_img_002.jpg
│   └── ...
└── error_analysis/                # 错误分析图像
    ├── fp_case_01_img_003.jpg     # 误检案例
    ├── fn_case_01_img_004.jpg     # 漏检案例
    └── ...
```

#### 评估结果文件示例（evaluation_results.json）
```json
{
  "overall_performance": {
    "mAP@0.5": 0.8542,
    "mAP@0.5:0.95": 0.6234,
    "Precision": 0.8912,
    "Recall": 0.7834,
    "F1-Score": 0.8339
  },
  "detection_counts": {
    "Total_TP": 156,
    "Total_FP": 19,
    "Total_FN": 43,
    "Total_GT": 199,
    "Total_Pred": 175
  },
  "confidence_analysis": {
    "avg_confidence": 0.7234,
    "avg_tp_confidence": 0.8123,
    "avg_fp_confidence": 0.4567,
    "best_conf_threshold": 0.6
  },
  "class_performance": {
    "class_0": {
      "mAP@0.5": 0.8542,
      "mAP@0.55": 0.8123,
      "mAP@0.6": 0.7834
    }
  }
}
```

#### 可视化图表说明

##### 1. 整体指标条形图（overall_metrics.png）
- 展示主要性能指标：mAP@0.5、mAP@0.5:0.95、Precision、Recall、F1-Score
- 每个指标显示具体数值和颜色编码
- 便于快速了解模型整体性能

##### 2. 检测结果分布图（detection_distribution.png）
- **左图**：TP、FP、FN的饼图分布，显示检测结果构成
- **右图**：总预测数vs总真实目标数对比柱状图
- 帮助理解模型的检测偏向性

##### 3. mAP vs IoU阈值曲线（map_vs_iou.png）
- 展示不同IoU阈值下的mAP变化趋势
- 标注关键点数值（0.5、0.75、0.9）
- 评估模型的定位精度能力

##### 4. 置信度分析图（confidence_distribution.png）
- **左图**：所有预测的置信度分布直方图
- **右图**：TP vs FP的置信度分布对比
- 分析模型的置信度校准情况

##### 5. 置信度vs性能曲线（confidence_vs_performance.png）
- 显示不同置信度阈值下的Precision、Recall、F1变化
- 标注最佳F1分数对应的置信度阈值
- 帮助选择最优的推理阈值

### 7.7 样本图像分析

#### 样本图像对比（sample_images/）
每个样本图像包含四个部分的可视化：
1. **Original**: 原始图像
2. **Predictions**: 预测结果（红色边界框）
3. **Ground Truth**: 真实标注（绿色边界框）
4. **Comparison**: 预测与真实对比（重叠分析）

#### 错误分析图像（error_analysis/）
- **FP Cases**: 高误检率图像案例分析
- **FN Cases**: 高漏检率图像案例分析
- 帮助识别模型的薄弱环节和改进方向

### 7.8 结果解读指南

#### 良好检测结果的特征
- **mAP@0.5 > 0.8**: 在标准IoU阈值下表现优秀
- **mAP@0.5:0.95 > 0.6**: 在严格阈值下仍有良好表现
- **Precision > 0.85**: 误检率低，检测可靠
- **Recall > 0.8**: 漏检率低，检测完整
- **TP置信度 > FP置信度**: 模型具有良好的区分能力

#### 需要改进的情况及建议
- **高FP率（低Precision）**: 
  - 提高置信度阈值
  - 增加负样本训练
  - 改进数据增强策略
- **高FN率（低Recall）**: 
  - 降低置信度阈值
  - 增加小目标样本
  - 调整锚框尺寸
- **mAP@0.5:0.95较低**: 
  - 改进边界框回归损失
  - 使用更精确的标注
  - 增加多尺度训练

### 7.9 高级使用技巧

#### 1. 小目标检测专用评估
```bash
# 使用更宽松的IoU阈值，适合小目标检测
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images \
    --iou-thresholds 0.3 0.4 0.5 0.6 0.7 \
    --name small_target_eval
```

#### 2. 批量评估多个模型
```bash
# 创建批量评估脚本
models=("yolov5s" "yolov5m" "yolov5l")
for model in "${models[@]}"; do
    python scripts/evaluate_detection.py \
        --pred-dir runs/detect/${model}/labels \
        --gt-dir data/labels \
        --img-dir data/images \
        --name eval_${model}
done
```

#### 3. 针对性错误分析
评估完成后，重点关注：
- 高FP图像：分析误检原因
- 高FN图像：分析漏检模式
- 低置信度TP：检查边界情况
- 高置信度FP：识别系统性错误

---

## 8. 评估结果对比功能（compare_evaluation_results.py）

### 8.1 功能概述

`compare_evaluation_results.py` 是多次评估结果对比分析的专用脚本，能够对比不同模型或不同配置下的检测效果，生成详细的对比报告和可视化图表。

#### 主要特性
- 📊 **多模型性能对比**：支持同时对比多个模型的性能指标
- 📈 **丰富的对比可视化**：提供条形图、雷达图、排名图等多种图表
- 🎯 **关键指标排名**：自动计算和展示各模型的排名情况
- 📋 **详细对比报告**：生成包含统计分析的文字报告
- 💾 **批量结果处理**：支持批量处理多个评估结果

### 8.2 使用方法

#### 基础对比命令
```bash
# 对比评估目录下的所有结果
python scripts/compare_evaluation_results.py --eval-dir runs/evaluation

# 指定输出目录
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --output-dir runs/comparison/model_comparison
```

#### 高级对比配置
```bash
# 对比指定的评估结果
python scripts/compare_evaluation_results.py \
    --eval-dirs runs/evaluation/model_v1 runs/evaluation/model_v2 runs/evaluation/model_v3 \
    --output-dir runs/comparison/three_models \
    --name model_comparison_v1

# 自定义对比指标和排序
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --metrics mAP@0.5 mAP@0.5:0.95 Precision Recall F1-Score \
    --sort-by mAP@0.5
```

### 8.3 输入要求

#### 评估结果目录结构
脚本需要读取由 `evaluate_detection.py` 生成的评估结果：
```
runs/evaluation/
├── yolov5s_experiment/
│   ├── evaluation_results.json    # 必需：评估结果数据
│   ├── config.yaml                # 必需：评估配置信息
│   └── *.png                      # 可选：可视化图表
├── yolov5m_experiment/
│   ├── evaluation_results.json
│   ├── config.yaml
│   └── *.png
└── yolov5l_experiment/
    ├── evaluation_results.json
    ├── config.yaml
    └── *.png
```

### 8.4 输出结果详解

#### 对比输出目录结构
```
runs/comparison/comparison/
├── config.yaml                    # 对比实验配置信息
├── comparison_results.json        # 详细对比数据（JSON格式）
├── summary_report.txt             # 文字对比报告
├── metrics_comparison.png         # 指标对比条形图
├── metrics_radar_chart.png        # 雷达图对比
├── ranking_chart.png              # 排名对比热力图
├── confidence_comparison.png      # 置信度分布对比
├── detailed_metrics_table.png     # 详细指标表格
└── performance_trend.png          # 性能趋势图
```

#### 主要图表功能说明

##### 1. 指标对比条形图（metrics_comparison.png）
- **功能**：并排展示所有模型在各指标上的表现
- **特点**：使用不同颜色区分模型，显示具体数值标签
- **用途**：快速识别各模型的优势指标

##### 2. 雷达图对比（metrics_radar_chart.png）
- **功能**：多维度性能对比的雷达图
- **特点**：直观展示模型的优势和劣势分布
- **用途**：适合比较模型的综合能力和平衡性

##### 3. 排名对比热力图（ranking_chart.png）
- **功能**：显示各模型在不同指标上的排名
- **特点**：热力图形式展示排名分布
- **用途**：便于识别性能一致性好的模型

##### 4. 置信度对比图（confidence_comparison.png）
- **功能**：对比各模型的置信度分布特征
- **特点**：展示TP和FP的置信度差异
- **用途**：分析模型的判别能力和校准情况

##### 5. 详细指标表格（detailed_metrics_table.png）
- **功能**：以表格形式展示所有详细指标
- **特点**：包含排名信息和高精度数值
- **用途**：便于精确数值比较和记录

### 8.5 结果解读指南

#### 优秀模型识别标准
- **mAP@0.5 > 0.8**: 基础检测性能优秀
- **mAP@0.5:0.95 > 0.6**: 定位精度优秀，适合严格应用
- **Precision和Recall均衡**: 避免严重偏向某一方面
- **置信度分离度高**: TP和FP置信度差异明显，模型判别能力强

#### 应用场景导向的模型选择
- **生产环境应用**: 优先选择Precision高的模型（减少误检）
- **研究和分析场景**: 优先选择Recall高的模型（减少漏检）
- **平衡性应用**: 选择F1-Score最高的模型（综合最优）
- **严格质量要求**: 选择mAP@0.5:0.95高的模型（定位精确）

#### 改进方向识别
通过对比分析识别模型弱点：
- **Precision普遍偏低**: 考虑提高置信度阈值或改进负样本训练
- **Recall普遍偏低**: 考虑降低置信度阈值或增加数据增强
- **mAP@0.5:0.95偏低**: 需要提高边界框定位精度，改进损失函数
- **性能差异大**: 说明超参数或数据对模型影响显著，需要优化

### 8.6 高级使用技巧

#### 1. 批量模型对比工作流
```bash
# 第一步：批量训练多个模型
models=("yolov5s" "yolov5m" "yolov5l")
for model in "${models[@]}"; do
    python scripts/start_training.py \
        --cfg models/${model}.yaml \
        --weights weights/${model}.pt \
        --name ${model}_experiment
done

# 第二步：批量评估模型
for model in "${models[@]}"; do
    python scripts/evaluate_detection.py \
        --pred-dir runs/detect/${model}_experiment/labels \
        --gt-dir data/labels \
        --img-dir data/images \
        --name eval_${model}
done

# 第三步：对比分析
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --name model_architecture_comparison
```

#### 2. 超参数对比分析
```bash
# 生成超参数对比图表
python scripts/generate_charts.py \
    --results \
        runs/small_target_train/conservative/results.csv \
        runs/small_target_train/standard/results.csv \
        runs/small_target_train/aggressive/results.csv \
    --labels "Conservative" "Standard" "Aggressive" \
    --output-dir runs/charts/hyperparameter_comparison \
    --name hyperparams_comparison
```

#### 3. 自定义指标权重分析
根据业务需求调整对比重点：
```bash
# 注重准确性的应用场景
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --metrics Precision mAP@0.5:0.95 F1-Score \
    --sort-by Precision

# 注重完整性的应用场景  
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --metrics Recall mAP@0.5 F1-Score \
    --sort-by Recall
```

---

## 9. 训练图表生成功能（generate_charts.py）

### 9.1 功能概述

`generate_charts.py` 是训练结果可视化的专用脚本，能够从训练的results.csv文件生成丰富的训练过程图表，帮助分析训练效果和调优参数。

#### 主要特性
- 📊 **多种训练曲线可视化**：损失曲线、性能指标、学习率变化等
- 📈 **损失函数趋势分析**：详细分析各损失组件的变化趋势
- 🎯 **性能指标变化图表**：Precision、Recall、mAP等指标的进展
- 📋 **训练统计信息汇总**：自动生成训练摘要报告
- 💾 **高质量图表输出**：支持多种格式和分辨率

### 9.2 使用方法

#### 基础图表生成
```bash
# 生成单个训练的图表
python scripts/generate_charts.py --results runs/small_target_train/exp/results.csv

# 指定输出目录
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv \
    --output-dir runs/charts/training_analysis
```

#### 多模型对比图表
```bash
# 对比多个训练结果
python scripts/generate_charts.py \
    --results runs/small_target_train/exp1/results.csv runs/small_target_train/exp2/results.csv \
    --labels "YOLOv5s" "YOLOv5m" \
    --output-dir runs/charts/model_comparison

# 自定义图表样式
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv \
    --style dark \
    --dpi 300 \
    --format png
```

### 9.3 输入文件要求

#### results.csv文件结构
训练脚本会自动生成包含以下列的CSV文件：
```csv
epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,val/box_loss,val/obj_loss,val/cls_loss,lr/pg0,lr/pg1,lr/pg2
0,0.05234,0.02145,0.00156,0.0,0.0,0.0,0.0,0.04987,0.02089,0.00134,0.01,0.01,0.01
1,0.04876,0.01987,0.00142,0.123,0.089,0.067,0.034,0.04654,0.01943,0.00128,0.0099,0.0099,0.0099
...
```

#### 关键数据列说明
- **epoch**: 训练轮次
- **train/box_loss**: 训练边界框损失
- **train/obj_loss**: 训练目标损失  
- **train/cls_loss**: 训练分类损失
- **metrics/precision**: 验证精确度
- **metrics/recall**: 验证召回率
- **metrics/mAP_0.5**: 验证mAP@0.5
- **metrics/mAP_0.5:0.95**: 验证mAP@0.5:0.95
- **val/box_loss**: 验证边界框损失
- **lr/pg0**: 学习率组0

### 9.4 输出图表详解

#### 图表输出目录结构
```
runs/charts/charts/
├── training_overview.png          # 训练总览图（四合一）
├── loss_curves.png               # 详细损失曲线图
├── metrics_curves.png            # 性能指标曲线
├── learning_rate_schedule.png    # 学习率调度图
├── loss_components.png           # 损失组件分析
├── convergence_analysis.png      # 收敛性分析
├── training_summary.txt          # 训练统计摘要
└── combined_analysis.png         # 综合分析图
```

#### 主要图表功能说明

##### 1. 训练总览图（training_overview.png）
**布局**：2×2子图布局，一目了然掌握训练全貌
- **左上**：总损失变化曲线（训练vs验证）
- **右上**：mAP@0.5和mAP@0.5:0.95变化趋势
- **左下**：精确度和召回率变化
- **右下**：学习率调度曲线

**用途**：快速评估训练效果和识别问题

##### 2. 损失曲线图（loss_curves.png）
- **训练vs验证损失对比**：识别过拟合/欠拟合
- **三个损失组件**：box_loss、obj_loss、cls_loss
- **趋势分析**：收敛速度和稳定性
- **关键点标注**：最低损失点和异常点

##### 3. 性能指标曲线（metrics_curves.png）
- **主要指标**：Precision、Recall、F1-Score、mAP
- **进展趋势**：性能提升的连续性
- **最佳点标注**：各指标的峰值点
- **收敛状态**：判断是否需要继续训练

##### 4. 学习率调度图（learning_rate_schedule.png）
- **多参数组显示**：不同层的学习率变化
- **策略可视化**：余弦退火、步长衰减等
- **性能关联**：学习率变化与性能提升同步

##### 5. 损失组件分析（loss_components.png）
- **相对贡献**：各损失组件的权重
- **收敛对比**：不同组件的收敛速度
- **问题诊断**：识别特定类型的训练问题

##### 6. 收敛性分析（convergence_analysis.png）
- **平滑趋势**：过滤噪声后的真实趋势
- **稳定性评估**：后期波动幅度分析
- **早停建议**：基于收敛状态的停止建议

---

## 10. 训练监控功能（training_monitor.py）

### 10.1 功能概述

`training_monitor.py` 是实时监控YOLOv5训练进程的可视化工具，提供训练状态追踪、性能分析和异常检测功能。

#### 主要特性
- 📊 **实时训练指标监控**：损失曲线、性能指标、学习率变化等实时更新
- 🔍 **异常检测和报警**：自动识别训练异常并发送告警通知
- 📈 **动态图表更新**：Web界面实时显示训练进展
- 💾 **训练日志分析**：深度分析训练数据和趋势
- 🎯 **性能瓶颈诊断**：识别并分析训练中的性能问题

### 10.2 使用方法

#### 基础监控命令
```bash
# 监控指定训练目录
python scripts/training_monitor.py --logdir runs/train/exp

# 自动检测并监控最新训练
python scripts/training_monitor.py --auto-detect

# 递归监控多个实验
python scripts/training_monitor.py --logdir runs/train --recursive
```

#### 高级监控配置
```bash
# 启用完整告警功能
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --alert \
    --email your@email.com \
    --threshold-loss 0.1 \
    --threshold-map 0.5 \
    --patience 20

# 自定义Web界面
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --refresh-interval 30 \
    --port 8080 \
    --theme dark \
    --host 0.0.0.0
```

### 10.3 监控指标体系

#### 核心训练指标
- **train/box_loss**: 边界框回归损失（定位精度）
- **train/obj_loss**: 目标性检测损失（检测置信度）
- **train/cls_loss**: 分类损失（类别识别）
- **train/total_loss**: 总训练损失（综合性能）

#### 验证性能指标
- **val/box_loss**: 验证边界框损失
- **val/obj_loss**: 验证目标性损失
- **val/cls_loss**: 验证分类损失
- **metrics/precision**: 精确度（误检控制）
- **metrics/recall**: 召回率（漏检控制）
- **metrics/mAP_0.5**: mAP@0.5（基础检测性能）
- **metrics/mAP_0.5:0.95**: mAP@0.5:0.95（严格检测性能）

#### 学习率监控
- **lr/pg0**: 骨干网络学习率
- **lr/pg1**: 颈部网络学习率
- **lr/pg2**: 检测头学习率

#### 系统监控指标
- **GPU利用率**: GPU使用百分比（理想值：>90%）
- **显存使用**: 显存占用情况（避免>95%）
- **训练速度**: 每秒处理图片数（img/s）
- **ETA**: 预计完成时间

### 10.4 Web界面功能

#### 访问监控界面
启动监控后，通过浏览器访问 `http://localhost:8888` 查看训练监控界面。

#### 主控制台布局

##### 1. 实时图表区域
```
损失曲线图表区:
├── 训练损失趋势图 (train_loss)
├── 验证损失趋势图 (val_loss)  
├── 总损失对比图 (total_loss)
└── 损失组件分解图 (loss_breakdown)

性能指标图表区:
├── mAP趋势图 (map_trend)
├── 精确度召回率图 (precision_recall)
├── F1分数变化图 (f1_score)
└── 类别AP对比图 (class_ap)

学习率图表区:
├── 学习率调度图 (lr_schedule)
├── 多组学习率图 (lr_groups)
└── 学习率vs性能关系图 (lr_vs_performance)
```

##### 2. 状态监控面板
```
训练状态显示:
├── 当前轮次/总轮次 (50/200)
├── 已训练时间 (2h 15m)
├── 预计剩余时间 (4h 30m)
├── 平均每轮用时 (2.5min)
└── 训练进度百分比 (25%)

硬件状态监控:
├── GPU温度 (76°C)
├── GPU利用率 (98.5%)
├── 显存使用率 (85.2%)
├── CPU使用率 (45%)
└── 内存使用率 (60%)

最新性能指标:
├── 当前总损失 (0.0234)
├── 最佳mAP@0.5 (0.798)
├── 最新精确度 (0.856)
├── 最新召回率 (0.743)
└── 最新F1分数 (0.795)
```

##### 3. 控制操作区
```
训练控制:
├── 暂停/恢复训练 [Pause/Resume]
├── 提前停止训练 [Early Stop]
├── 手动保存检查点 [Save Checkpoint]
└── 动态调整学习率 [Adjust LR]

数据导出:
├── 导出训练数据 [Export Data]
├── 保存当前图表 [Save Charts]
├── 生成训练报告 [Generate Report]
└── 下载最佳模型 [Download Model]
```

#### 对比分析页面
访问 `http://localhost:8888/compare` 进行多实验对比分析：

```
实验选择区:
├── 选择对比实验 (复选框列表)
├── 设置对比指标 (mAP, Loss, etc.)
├── 选择时间范围 (最近N轮)
└── 应用过滤条件 (性能阈值)

对比图表区:
├── 多实验损失对比图
├── 性能指标对比图
├── 训练效率对比图
└── 收敛速度分析图

统计分析区:
├── 最终性能排名表
├── 收敛时间统计图
├── 稳定性分析图
└── 资源消耗对比表
```

### 10.5 智能告警机制

#### 性能异常告警
脚本会自动检测以下异常情况：

##### 训练性能告警
- **损失异常上升**: 训练损失连续5轮上升
- **性能停滞**: mAP超过patience轮（默认20轮）无改善
- **过拟合检测**: 验证损失/训练损失 > 1.5
- **收敛异常**: 学习率过低但损失仍高

##### 系统资源告警
- **GPU温度过高**: 超过80°C时发出警告
- **显存不足**: 显存使用率超过95%
- **磁盘空间不足**: 磁盘使用率超过90%
- **训练速度过慢**: 每秒处理图片数低于阈值

#### 告警配置示例
```bash
# 配置邮件告警
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export EMAIL_USER="your_email@gmail.com"
export EMAIL_PASS="your_app_password"

# 启用完整告警功能
python scripts/training_monitor.py \
    --logdir runs/train/exp \
    --alert \
    --email recipient@email.com \
    --threshold-loss 0.1 \
    --threshold-map 0.5 \
    --patience 20 \
    --email-interval 300
```

### 10.6 输出文件详解

#### 监控数据目录结构
```
monitoring_output/
├── real_time_data/              # 实时监控数据
│   ├── training_metrics.json   # 训练指标时序数据
│   ├── system_stats.json       # 系统状态数据
│   └── alerts.log              # 告警事件日志
```

---

## 11. 图形界面测试工具（gui_test.py）

### 11.1 功能概述

`gui_test.py` 是一个功能完整的图形界面测试工具，提供直观的YOLOv5小目标检测模型测试界面，支持参数配置、实时日志显示和TensorRT加速。

#### 主要特性
- 🖥️ **直观的图形用户界面**：无需命令行操作，适合非技术用户
- 🔧 **完整的参数配置选项**：所有检测参数都可通过界面设置
- 📊 **实时训练/测试日志显示**：实时查看处理进度和结果
- ⚡ **TensorRT引擎生成和加速**：支持一键生成TensorRT引擎
- 🎯 **多模式检测支持**：图片、视频和批量检测
- 💾 **配置保存和加载功能**：方便重复使用常用设置

### 11.2 使用方法详解

#### 启动图形界面
```bash
# 启动GUI应用
python scripts/gui_test.py

# 后台运行（Linux/Mac）
nohup python scripts/gui_test.py &
```

#### 基本操作流程
1. **选择权重文件**：点击"浏览"按钮选择模型权重文件（.pt或.engine）
2. **选择测试源**：根据测试模式选择图片目录或视频目录
3. **配置检测参数**：调整置信度、IoU阈值等参数
4. **设置输出选项**：选择需要保存的结果类型
5. **启动测试**：点击"开始测试"按钮
6. **查看结果**：在日志区域查看实时输出和处理结果

### 11.3 界面组件详解

#### 文件选择区域
| 组件 | 功能 | 支持格式 | 默认值 |
|------|------|----------|--------|
| **权重文件** | 选择模型权重 | `.pt`, `.engine` | 无 |
| **测试源** | 选择测试数据 | 目录路径 | 无 |
| **输出目录** | 设置结果保存路径 | 目录路径 | `runs/detect/gui_test` |

#### 测试模式选择
| 模式 | 说明 | 适用场景 | 输出结果 |
|------|------|----------|----------|
| **视频检测** | 检测视频文件中的目标 | 单个或多个视频文件 | 标注视频、标签文件、裁剪图像 |
| **图片检测** | 检测图片中的目标 | 图片目录批量处理 | 标注图片、标签文件、裁剪图像 |
| **批量检测** | 批量处理多种格式文件 | 混合格式数据处理 | 按格式分类输出结果 |

#### 核心参数配置
| 参数 | 控件类型 | 取值范围 | 默认值 | 说明 |
|------|----------|----------|--------|------|
| **置信度阈值** | 滑块 | 0.01-1.0 | 0.25 | 检测置信度门槛，越高误检越少 |
| **IOU阈值** | 滑块 | 0.01-1.0 | 0.45 | NMS去重阈值，越高重复检测越多 |
| **图像尺寸** | 下拉框 | 320,416,512,640,832,1280 | 640 | 输入图像尺寸，越大精度越高但速度越慢 |
| **推理设备** | 下拉框 | cpu,0,1,0,1 | 0 | 推理设备选择（0为第一块GPU） |
| **最大检测数** | 数值框 | 1-10000 | 1000 | 单张图片最大检测目标数量 |

#### 输出控制参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save-txt` | 标志 | False | 保存YOLO格式检测标签 |
| `--save-conf` | 标志 | False | 在标签中包含置信度信息 |
| `--save-crop` | 标志 | False | 保存检测目标的裁剪图像 |
| `--save-as-images` | 标志 | False | 将结果保存为单独的图片文件 |
| `--nosave` | 标志 | False | 不保存检测后的图像/视频 |
| `--view-img` | 标志 | False | 实时显示检测结果 |

#### 性能优化参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--half` | 标志 | False | 使用FP16半精度推理（提高速度） |
| `--vid-stride` | 整数 | 1 | 视频帧采样步长（>1跳帧处理） |
| `--max-frames` | 整数 | 3000 | 最大处理帧数限制 |
| `--augment` | 标志 | False | 启用测试时间增强（TTA） |

#### 时间和区域控制参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start-time` | 字符串 | None | 开始时间（HH:MM:SS格式） |
| `--end-time` | 字符串 | None | 结束时间（HH:MM:SS格式） |
| `--roi-params` | 浮点数列表 | None | ROI参数：x y width height |
| `--roi-mode` | 字符串 | 'percentage' | ROI模式（'percentage'或'absolute'） |

#### 显示控制参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--line-thickness` | 整数 | 3 | 检测框线条粗细（像素） |
| `--hide-labels` | 标志 | False | 隐藏类别标签 |
| `--hide-conf` | 标志 | False | 隐藏置信度数值 |

### 11.4 TensorRT加速功能

#### TensorRT加速概述
TensorRT是NVIDIA提供的深度学习推理优化库，可以显著提升推理速度。

| 功能 | 说明 | 使用条件 | 性能提升 |
|----------|------|----------|----------|
| **启用TensorRT加速** | 使用优化后的TensorRT引擎推理 | NVIDIA GPU + TensorRT环境 | 2-5倍速度提升 |
| **生成TensorRT引擎** | 从PyTorch权重转换生成引擎 | 首次使用时需要 | 一次生成，后续重复使用 |

#### TensorRT使用步骤
1. **环境检查**：确保已安装TensorRT和CUDA
2. **生成引擎**：在GUI中勾选"启用TensorRT加速"
3. **一键转换**：点击"生成TensorRT引擎"按钮
4. **等待完成**：首次生成需要几分钟时间
5. **后续使用**：生成的.engine文件可重复使用

#### 性能对比参考
| 推理方式 | 相对速度 | 精度影响 | 显存占用 | 适用场景 |
|----------|----------|----------|----------|----------|
| **PyTorch FP32** | 1x（基准） | 无影响 | 高 | 精度要求极高 |
| **PyTorch FP16** | 1.5-2x | 微小下降 | 中等 | 平衡精度与速度 |
| **TensorRT FP32** | 2-3x | 无影响 | 中等 | 高精度实时应用 |
| **TensorRT FP16** | 3-5x | 微小下降 | 低 | 高速实时应用 |

### 11.5 输出结果详解

#### 检测结果目录结构
```
runs/detect/gui_test/exp/
├── labels/                     # YOLO格式标签文件
│   ├── image1.txt             # 图片1的检测标签
│   ├── image2.txt             # 图片2的检测标签
│   ├── video1_frame001.txt    # 视频帧标签
│   └── ...

├── crops/                      # 裁剪的目标图像
│     └── target/                # 按类别分组
│       ├── image1_target_1.jpg # 裁剪的目标1
│       ├── image1_target_2.jpg # 裁剪的目标2
│       └── ...
├── image1.jpg                  # 标注后的原图1
├── image2.jpg                  # 标注后的原图2
├── video1.mp4                  # 标注后的视频1
├── results.txt                 # 检测统计信息汇总
└── speed_stats.txt             # 处理速度统计
```

#### 标签文件格式详解
标签文件采用YOLO格式，每行一个检测目标：
```
# 格式: class_id x_center y_center width height [confidence]
# 示例：
0 0.5 0.3 0.1 0.2 0.85    # 类别0，中心(0.5,0.3)，尺寸(0.1×0.2)，置信度0.85
0 0.7 0.6 0.15 0.25 0.92  # 另一个目标
```

#### 实时日志信息
GUI底部的日志区域会显示详细的处理信息：
```
[14:30:15] 🚀 开始YOLOv5小目标检测测试
[14:30:15] 📁 权重文件: runs/small_target_train/exp/weights/best.pt
[14:30:15] 📁 测试源: /home/user/test_videos/
[14:30:15] 📁 测试模式: video (视频检测)
[14:30:15] ⚙️  参数配置: conf=0.25, iou=0.45, imgsz=640
[14:30:16] ⚡ 模型加载完成 (512.3ms)
[14:30:16] 🎯 开始处理: video1.mp4 (1920×1080, 30fps)
[14:30:18] ✅ 检测完成: 发现 15 个目标 (平均置信度: 0.78)
[14:30:18] 💾 结果保存至: runs/detect/gui_test/exp
[14:30:18] 📊 处理统计: 300帧, 速度: 45.6 FPS
```

### 11.6 配置文件管理

#### 配置保存功能
GUI支持将当前所有参数设置保存为JSON配置文件：
```json
{
  "file_paths": {
    "weights_path": "runs/small_target_train/exp/weights/best.pt",
    "source_path": "/home/user/test_videos/",
    "output_path": "runs/detect/gui_test"
  },
  "detection_params": {
    "test_mode": "video",
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "max_det": 1000,
    "imgsz": 640,
    "device": "0"
  },
  "output_options": {
    "save_txt": true,
    "save_conf": true,
    "save_crop": true,
    "save_as_images": false,
    "nosave": false,
    "view_img": false,
    "half": true,
    "augment": false
  },
  "advanced_settings": {
    "max_frames": 300,
    "vid_stride": 1,
    "line_thickness": 3,
    "time_range_enabled": false,
    "start_time": "00:00:00",
    "end_time": "00:01:00"
  },
  "tensorrt_settings": {
    "use_tensorrt": false,
    "fp16": true
  }
}
```

#### 配置加载功能
- 点击"加载配置"按钮可以导入之前保存的配置
- 所有参数将自动恢复到保存时的状态
- 支持多套配置方案，适应不同检测场景

### 11.7 使用技巧和最佳实践

#### 参数调优策略
1. **置信度阈值调整**：
   - 提高（0.3→0.5）：减少误检，但可能增加漏检
   - 降低（0.25→0.15）：减少漏检，但可能增加误检
   - 建议：先用0.25测试，根据结果调整

2. **IoU阈值优化**：
   - 提高（0.45→0.6）：减少重复检测，适合密集目标
   - 降低（0.45→0.3）：保留更多检测，适合稀疏目标

3. **图像尺寸选择**：
   - 小目标检测推荐832或1280
   - 实时应用可用640或512
   - 精度要求高时使用1280

#### 处理速度优化
1. **硬件优化**：
   - 使用NVIDIA GPU（推荐RTX 3060以上）
   - 确保足够显存（8GB+）
   - 使用SSD存储提高I/O速度

2. **软件优化**：
   - 启用TensorRT加速（首选）
   - 使用半精度推理
   - 适当降低图像尺寸
   - 增加视频步幅（仅视频检测）

3. **参数优化**：
   - 减少最大检测数（1000→500）
   - 关闭"显示结果"选项
   - 禁用数据增强（实时应用）

#### 质量控制方法
1. **结果验证**：
   - 开启"保存裁剪图像"检查检测质量
   - 查看标注图像确认检测效果
   - 分析置信度分布判断模型表现

2. **批量处理前测试**：
   - 先用少量数据测试参数效果
   - 确认检测质量后再大批量处理
   - 保存最优参数配置供后续使用

### 11.8 故障排除

#### 常见问题及解决方案

##### GUI启动失败
```bash
# 问题：ImportError: No module named 'tkinter'
# 解决方案：
# Ubuntu/Debian系统：
sudo apt-get install python3-tk

# CentOS/RHEL系统：
sudo yum install tkinter

# Conda环境：
conda install tk
```

##### 模型加载失败
```bash
# 问题：权重文件损坏或路径错误
# 解决方案：
# 1. 检查文件路径是否正确
# 2. 验证权重文件完整性
python -c "import torch; torch.load('path/to/weights.pt')"
# 3. 重新下载或训练模型
```

##### TensorRT转换失败
```bash
# 问题：TensorRT环境配置问题
# 解决方案：
# 1. 检查CUDA版本兼容性
nvidia-smi

# 2. 验证TensorRT安装
python -c "import tensorrt; print(tensorrt.__version__)"

# 3. 检查GPU显存是否足够（推荐8GB+）
```

##### 检测效果不佳
**症状**：误检率高或漏检严重
**解决方案**：
1. 调整置信度阈值（提高减少误检，降低减少漏检）
2. 检查模型是否适合当前检测场景
3. 尝试不同的图像尺寸设置
4. 确认输入数据质量是否良好

##### 处理速度过慢
**症状**：检测速度无法满足需求
**解决方案**：
1. 启用TensorRT加速（优先推荐）
2. 开启半精度推理
3. 降低输入图像尺寸
4. 增加视频处理步幅
5. 减少最大检测数量

---

## 12. 标签类别修改工具（update_labels_class.py）

### 12.1 功能概述

`update_labels_class.py` 是一个专门用于批量修改YOLO格式标注文件类别ID的实用工具，主要用于将多类别检测任务转换为单类别小目标检测任务。

#### 主要特性
- 🔄 **批量处理**：支持一次性处理多个目录下的所有标注文件
- 💾 **自动备份**：处理前自动创建备份文件，确保数据安全
- 🎯 **单类别转换**：将所有类别ID统一改为0（单类别检测）
- 📁 **递归搜索**：自动搜索子目录中的所有.txt标注文件
- ✅ **格式验证**：验证YOLO标注格式的完整性
- 🖥️ **交互模式**：支持命令行交互和自动检测常见目录

### 12.2 使用场景

#### 典型应用场景
1. **多类别转单类别**：将COCO等多类别数据集转换为小目标单类别检测
2. **数据集整合**：合并不同来源的标注数据，统一类别标识
3. **模型简化**：将复杂的多类别模型简化为单类别检测模型
4. **标注错误修复**：批量修正标注文件中的类别ID错误

#### 处理前后对比
```bash
# 处理前的标注文件内容：
1 0.5 0.3 0.1 0.2    # 类别1：人员
2 0.7 0.6 0.15 0.25  # 类别2：车辆
3 0.2 0.8 0.08 0.12  # 类别3：飞机

# 处理后的标注文件内容：
0 0.5 0.3 0.1 0.2    # 统一类别0：小目标
0 0.7 0.6 0.15 0.25  # 统一类别0：小目标  
0 0.2 0.8 0.08 0.12  # 统一类别0：小目标
```

### 12.3 使用方法详解

#### 基本命令格式
```bash
# 基本用法：处理单个目录
python scripts/update_labels_class.py --dirs /path/to/labels

# 处理多个目录
python scripts/update_labels_class.py --dirs /path/to/labels1 /path/to/labels2 /path/to/labels3

# 不创建备份文件（谨慎使用）
python scripts/update_labels_class.py --dirs /path/to/labels --no-backup
```

#### 交互模式使用
```bash
# 直接运行脚本，自动检测常见目录
python scripts/update_labels_class.py

# 输出示例：
🔍 检测到可能的labels目录...
找到以下labels目录
```
