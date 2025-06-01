# generate_charts.py - 训练图表生成脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/generate_charts.py`

## 📋 功能概述

这是训练结果可视化的专用脚本，能够从训练的results.csv文件生成丰富的训练过程图表，帮助分析训练效果和调优参数。

### 主要特性
- 📊 多种训练曲线可视化
- 📈 损失函数趋势分析
- 🎯 性能指标变化图表
- 📋 训练统计信息汇总
- 💾 高质量图表输出

## 🚀 使用方法

### 基础使用
```bash
# 生成训练图表
python scripts/generate_charts.py --results runs/small_target_train/exp/results.csv

# 指定输出目录
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv \
    --output-dir runs/charts/training_analysis
```

### 高级使用
```bash
# 对比多个训练结果
python scripts/generate_charts.py \
    --results runs/small_target_train/exp1/results.csv runs/small_target_train/exp2/results.csv \
    --labels "Model_V1" "Model_V2" \
    --output-dir runs/charts/model_comparison

# 自定义图表样式
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv \
    --style dark \
    --dpi 300 \
    --format png
```

## 📝 参数详解

### 必需参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--results` | str/list | 无 | 训练结果CSV文件路径 | 命令行 |

### 可选参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--output-dir` | str | `runs/charts` | 图表输出目录 | 命令行 |
| `--labels` | list | 无 | 多结果对比时的标签 | 命令行 |
| `--name` | str | `charts` | 图表集合名称 | 命令行 |

### 图表配置参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--style` | str | `seaborn` | 图表样式主题 | 命令行 |
| `--dpi` | int | `300` | 图表分辨率 | 命令行 |
| `--format` | str | `png` | 输出格式(png/pdf/svg) | 命令行 |
| `--figsize` | tuple | `(12, 8)` | 图表尺寸 | 脚本内修改 |

## 📂 输入文件格式

### results.csv文件结构
```csv
epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,val/box_loss,val/obj_loss,val/cls_loss,lr/pg0,lr/pg1,lr/pg2
0,0.05234,0.02145,0.00156,0.0,0.0,0.0,0.0,0.04987,0.02089,0.00134,0.01,0.01,0.01
1,0.04876,0.01987,0.00142,0.123,0.089,0.067,0.034,0.04654,0.01943,0.00128,0.0099,0.0099,0.0099
...
```

### 必需的列字段
- **epoch**: 训练轮次
- **train/box_loss**: 训练边界框损失
- **train/obj_loss**: 训练目标损失
- **train/cls_loss**: 训练分类损失
- **metrics/precision**: 验证精确度
- **metrics/recall**: 验证召回率
- **metrics/mAP_0.5**: 验证mAP@0.5
- **metrics/mAP_0.5:0.95**: 验证mAP@0.5:0.95
- **val/box_loss**: 验证边界框损失
- **val/obj_loss**: 验证目标损失
- **val/cls_loss**: 验证分类损失
- **lr/pg0**: 学习率组0

## 📊 输出文件说明

### 图表输出目录结构
```
runs/charts/charts/
├── training_overview.png          # 训练总览图
├── loss_curves.png               # 损失曲线图
├── metrics_curves.png            # 性能指标曲线
├── learning_rate_schedule.png    # 学习率调度图
├── loss_components.png           # 损失组件分析
├── convergence_analysis.png      # 收敛性分析
├── training_summary.txt          # 训练统计摘要
└── combined_analysis.png         # 综合分析图
```

### 主要图表说明

#### 1. 训练总览图 (training_overview.png)
- **2×2子图布局**，包含主要训练指标
- **左上**: 总损失变化曲线
- **右上**: mAP@0.5和mAP@0.5:0.95变化
- **左下**: 精确度和召回率变化
- **右下**: 学习率调度曲线

#### 2. 损失曲线图 (loss_curves.png)
- **训练vs验证损失对比**
- 包含box_loss、obj_loss、cls_loss三个组件
- 显示过拟合/欠拟合趋势
- 标注最低损失点

#### 3. 性能指标曲线 (metrics_curves.png)
- **Precision、Recall、F1-Score**变化趋势
- **mAP@0.5和mAP@0.5:0.95**进展
- 标注最佳性能点
- 显示收敛状态

#### 4. 学习率调度图 (learning_rate_schedule.png)
- 不同参数组的学习率变化
- 余弦退火/步长衰减等策略可视化
- 与性能变化的关联分析

#### 5. 损失组件分析 (loss_components.png)
- 三个损失组件的相对贡献
- 各组件的收敛速度对比
- 帮助诊断训练问题

#### 6. 收敛性分析 (convergence_analysis.png)
- 损失和指标的平滑趋势
- 训练稳定性评估
- 早停建议点标注

## 🎯 图表解读指南

### 良好训练的特征
- **损失曲线**: 平滑下降，训练与验证损失接近
- **性能指标**: 稳步上升并趋于稳定
- **学习率**: 合理的衰减策略
- **收敛性**: 后期波动较小

### 常见问题识别

#### 过拟合信号
- 训练损失持续下降，验证损失上升
- 训练精度高，验证精度低
- 验证指标波动剧烈

#### 欠拟合信号
- 训练和验证损失都很高
- 性能指标提升缓慢
- 损失下降平缓

#### 学习率问题
- 学习率过高：损失震荡剧烈
- 学习率过低：收敛速度慢
- 需要调整：后期仍有下降空间

## 📈 训练统计摘要 (training_summary.txt)
```
============================================================
训练统计摘要报告
============================================================
训练配置:
- 总轮次: 200
- 最佳轮次: 156
- 收敛轮次: 145

性能指标:
- 最佳 mAP@0.5: 0.8542 (轮次 156)
- 最佳 mAP@0.5:0.95: 0.6234 (轮次 154)
- 最佳 Precision: 0.8912 (轮次 158)
- 最佳 Recall: 0.8234 (轮次 152)

损失分析:
- 最低训练损失: 0.0234 (轮次 198)
- 最低验证损失: 0.0267 (轮次 156)
- 过拟合程度: 轻微 (差值 0.0033)

学习率:
- 初始学习率: 0.01
- 最终学习率: 0.0001
- 衰减策略: 余弦退火

训练建议:
- 训练效果良好，收敛稳定
- 可尝试延长训练以进一步优化
- 建议保存轮次156的权重作为最佳模型
============================================================
```

## ⚠️ 常见问题和解决方案

### 1. CSV文件格式错误
```bash
# 问题：CSV文件缺少必要列或格式不正确
# 解决方案：检查results.csv文件结构
head -n 2 runs/small_target_train/exp/results.csv

# 确保包含所有必需列
```

### 2. 图表显示异常
```bash
# 问题：图表中文显示异常或布局错乱
# 解决方案：脚本已优化中文字体支持，检查系统字体
fc-list :lang=zh

# 如果仍有问题，可以修改脚本中的字体设置
```

### 3. 内存不足
```bash
# 问题：处理大型训练结果时内存不足
# 解决方案：减少图表分辨率或分批处理
python scripts/generate_charts.py --results ... --dpi 150
```

### 4. 多结果对比问题
```bash
# 问题：对比多个结果时标签不匹配
# 解决方案：确保results文件数量与labels数量一致
python scripts/generate_charts.py \
    --results file1.csv file2.csv \
    --labels "Label1" "Label2"
```

## 🔧 高级使用技巧

### 1. 批量生成图表
```bash
# 为多个训练实验生成图表
for exp in exp1 exp2 exp3; do
    python scripts/generate_charts.py \
        --results runs/small_target_train/${exp}/results.csv \
        --output-dir runs/charts/${exp}_analysis \
        --name ${exp}_charts
done
```

### 2. 自定义图表样式
```python
# 在脚本中可以修改的样式选项
CHART_STYLES = {
    'seaborn': 'seaborn-v0_8',
    'dark': 'dark_background',
    'classic': 'classic',
    'bmh': 'bmh'
}

# 颜色主题
COLOR_PALETTES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    'colorblind': ['#0173b2', '#de8f05', '#029e73', '#cc78bc'],
    'pastel': ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
}
```

### 3. 导出高质量图表
```bash
# 生成出版质量的图表
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv \
    --dpi 600 \
    --format pdf \
    --style classic
```

## 📊 性能分析方法

### 1. 训练效率分析
- **收敛速度**: 达到90%最佳性能的轮次
- **训练稳定性**: 后期指标波动幅度
- **资源利用**: 训练时间与性能提升的关系

### 2. 超参数效果评估
- **学习率影响**: 不同学习率策略的效果对比
- **损失权重**: 各损失组件对总体性能的贡献
- **数据增强**: 增强策略对收敛的影响

### 3. 模型性能预测
- **早停预测**: 基于验证损失趋势判断最佳停止点
- **进一步训练**: 判断是否有继续训练的潜力
- **超参数调优**: 基于图表分析提出调优建议

## 🔗 脚本集成使用

### 与训练脚本结合
```bash
# 训练完成后自动生成图表
python scripts/small_target_train.py --data dataset.yaml --epochs 200
python scripts/generate_charts.py --results runs/small_target_train/exp/results.csv
```

### 与评估脚本结合
```bash
# 生成训练图表后进行模型评估
python scripts/generate_charts.py --results runs/small_target_train/exp/results.csv
python scripts/evaluate_detection.py --pred-dir runs/detect/exp/labels
```

## 🔗 相关文档

- [small_target_train.md](small_target_train.md) - 训练脚本使用
- [training_monitor.md](training_monitor.md) - 实时训练监控
- [compare_evaluation_results.md](compare_evaluation_results.md) - 多模型对比