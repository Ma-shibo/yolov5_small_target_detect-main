# evaluate_detection.py - 检测结果评估脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/evaluate_detection.py`

## 📋 功能概述

这是检测结果与人工标注对比分析的专用脚本，能够计算多种检测准确性指标，支持自动实验目录生成和丰富的可视化功能。

### 主要特性
- 🎯 多IoU阈值评估（mAP@0.5, mAP@0.5:0.95等）
- 📊 详细的置信度分析和统计
- 🖼️ 自动生成样本图像对比
- 📈 丰富的可视化图表
- 💾 自动实验目录管理

## 🚀 使用方法

### 基础使用
```bash
# 评估检测结果
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir /path/to/ground_truth/labels \
    --img-dir /path/to/images

# 指定输出目录
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images \
    --output-dir runs/evaluation/custom_eval
```

### 高级使用
```bash
# 自定义IoU阈值
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images \
    --iou-thresholds 0.3 0.5 0.7 0.9

# 指定实验名称和项目目录
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/exp/labels \
    --gt-dir data/labels \
    --img-dir data/images \
    --project runs/my_evaluation \
    --name test_model_v1
```

## 📝 参数详解

### 必需参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--pred-dir` | str | 无 | 预测标签文件目录 | 命令行 |
| `--gt-dir` | str | 无 | 真实标签文件目录 | 命令行 |
| `--img-dir` | str | 无 | 图片文件目录 | 命令行 |

### 可选参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--output-dir` | str | `runs/evaluation` | 结果输出目录 | 命令行 |
| `--project` | str | `runs/evaluation` | 项目根目录 | 命令行 |
| `--name` | str | `eval` | 实验名称 | 命令行 |

### 评估配置参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--iou-thresholds` | list | `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]` | IoU阈值列表 | 命令行 |

## 📂 输入文件格式要求

### 1. 预测标签文件格式 (YOLO格式)
```
# 文件名: image_name.txt
# 格式: class_id x_center y_center width height confidence
0 0.5 0.3 0.1 0.2 0.85
0 0.7 0.6 0.15 0.25 0.92
```

### 2. 真实标签文件格式 (YOLO格式)
```
# 文件名: image_name.txt  
# 格式: class_id x_center y_center width height
0 0.5 0.3 0.1 0.2
0 0.7 0.6 0.15 0.25
```

### 3. 图片文件要求
- 支持格式: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- 文件命名必须与标签文件对应 (除扩展名外)

### 4. 目录结构示例
```
project/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── ground_truth_labels/
│   ├── img_001.txt
│   ├── img_002.txt  
│   └── ...
└── prediction_labels/
    ├── img_001.txt
    ├── img_002.txt
    └── ...
```

## 📊 输出文件说明

### 评估输出目录结构
```
runs/evaluation/eval/
├── config.yaml                    # 实验配置信息
├── evaluation_results.json        # 详细评估结果
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
    ├── fp_case_01_img_003.jpg
    ├── fn_case_01_img_004.jpg
    └── ...
```

### 主要输出文件解释

#### 1. 评估结果文件 (evaluation_results.json)
```json
{
  "mAP@0.5": 0.8542,
  "mAP@0.5:0.95": 0.6234,
  "Precision": 0.8912,
  "Recall": 0.7834,
  "F1-Score": 0.8339,
  "Total_TP": 156,
  "Total_FP": 19,
  "Total_FN": 43,
  "Total_GT": 199,
  "Total_Pred": 175,
  "confidence_stats": {
    "avg_confidence": 0.7234,
    "avg_tp_confidence": 0.8123,
    "avg_fp_confidence": 0.4567
  },
  "class_aps": {
    "0": {
      "0.5": 0.8542,
      "0.55": 0.8123,
      "0.6": 0.7834
    }
  }
}
```

#### 2. 配置信息文件 (config.yaml)
```yaml
experiment_info:
  start_time: '2025-05-28 14:30:15'
  script_path: '/path/to/evaluate_detection.py'
  working_directory: '/path/to/project'
data_paths:
  pred_dir: 'runs/detect/exp/labels'
  gt_dir: 'data/labels'
  img_dir: 'data/images'
  output_dir: 'runs/evaluation/eval'
parameters:
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
  name: 'eval'
  project: 'runs/evaluation'
```

## 🎯 评估指标详解

### 基础检测指标
- **TP (True Positive)**: 正确检测的目标数量
- **FP (False Positive)**: 错误检测的数量（误检）
- **FN (False Negative)**: 漏检的目标数量
- **TN (True Negative)**: 正确识别的背景区域

### 性能评估指标
- **Precision**: 精确度 = TP/(TP+FP)，表示检测准确性
- **Recall**: 召回率 = TP/(TP+FN)，表示检测完整性  
- **F1-Score**: F1分数 = 2×(Precision×Recall)/(Precision+Recall)，综合指标
- **mAP@0.5**: IoU阈值0.5时的平均精度
- **mAP@0.5:0.95**: IoU阈值0.5到0.95的平均精度

### 置信度相关指标
- **平均置信度**: 所有预测的平均置信度分数
- **TP平均置信度**: 正确检测的平均置信度
- **FP平均置信度**: 误检的平均置信度
- **最佳置信度阈值**: F1分数最高时对应的置信度阈值

## 📈 可视化图表说明

### 1. 整体指标条形图 (overall_metrics.png)
- 展示mAP@0.5、mAP@0.5:0.95、Precision、Recall、F1-Score
- 每个指标显示具体数值
- 使用不同颜色区分各指标

### 2. 检测结果分布图 (detection_distribution.png)
- 左图：TP、FP、FN的饼图分布
- 右图：总预测数vs总真实目标数对比

### 3. 各类别AP对比图 (class_ap_comparison.png)
- 比较不同类别的AP@0.5和AP@0.5:0.95
- 适用于多类别检测任务

### 4. mAP vs IoU阈值曲线 (map_vs_iou.png)
- 展示不同IoU阈值下的mAP变化
- 标注关键点数值（0.5、0.75、0.9）

### 5. 置信度分析图 (confidence_distribution.png)
- 左图：所有预测的置信度分布直方图
- 右图：TP vs FP的置信度分布对比

### 6. 置信度vs性能曲线 (confidence_vs_performance.png)
- 显示不同置信度阈值下的Precision、Recall、F1变化
- 标注最佳F1分数对应的置信度阈值

## 🖼️ 样本图像分析

### 样本图像对比 (sample_images/)
每个样本图像包含四个部分：
1. **Original**: 原始图像
2. **Predictions**: 预测结果（红色框）
3. **Ground Truth**: 真实标注（绿色框）
4. **Comparison**: 预测与真实对比

### 错误分析图像 (error_analysis/)
- **FP Cases**: 高误检率图像案例
- **FN Cases**: 高漏检率图像案例
- 帮助分析模型的薄弱环节

## ⚠️ 常见问题和解决方案

### 1. 找不到匹配文件
```bash
# 问题：没有找到匹配的预测和真实标签文件
# 解决方案：检查文件命名格式
ls /path/to/pred/labels/    # 检查预测文件
ls /path/to/gt/labels/      # 检查真实文件
# 确保文件名一致（除扩展名外）
```

### 2. 标签格式错误
```bash
# 问题：标签文件格式不正确
# 解决方案：检查YOLO格式要求
# 预测文件应包含置信度：class x_center y_center width height confidence
# 真实文件格式：class x_center y_center width height
```

### 3. 图像尺寸问题
```bash
# 问题：坐标超出图像范围
# 解决方案：检查标注质量
python scripts/update_labels_class.py --check-bounds
```

### 4. 内存不足
```bash
# 问题：处理大量图像时内存不足
# 解决方案：分批处理或使用更小的样本数量
# 修改脚本中的max_samples参数
```

## 📊 结果解读指南

### 良好的检测结果特征
- **mAP@0.5 > 0.8**: 在IoU=0.5阈值下表现优秀
- **mAP@0.5:0.95 > 0.6**: 在严格阈值下仍有好表现
- **Precision > 0.85**: 误检率低
- **Recall > 0.8**: 漏检率低
- **TP置信度 > FP置信度**: 模型区分能力强

### 需要改进的情况
- **高FP率**: 考虑提高置信度阈值或改进训练
- **高FN率**: 考虑降低置信度阈值或增加数据增强
- **mAP@0.5:0.95较低**: 边界框定位精度需提升

## 🔧 高级使用技巧

### 1. 批量评估多个模型
```bash
# 创建批量评估脚本
for model in model1 model2 model3; do
    python scripts/evaluate_detection.py \
        --pred-dir runs/detect/${model}/labels \
        --gt-dir data/labels \
        --img-dir data/images \
        --name eval_${model}
done
```

### 2. 自定义IoU阈值
```bash
# 针对小目标检测使用更宽松的IoU阈值
python scripts/evaluate_detection.py \
    --iou-thresholds 0.3 0.4 0.5 0.6 0.7
```

### 3. 结合对比分析
```bash
# 先评估多个模型，再对比结果
python scripts/compare_evaluation_results.py --eval-dir runs/evaluation
```

## 🔗 相关文档

- [compare_evaluation_results.md](compare_evaluation_results.md) - 多模型结果对比
- [small_target_train.md](small_target_train.md) - 模型训练
- [gui_test.md](gui_test.md) - 快速测试工具