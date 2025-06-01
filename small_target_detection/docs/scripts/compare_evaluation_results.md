# compare_evaluation_results.py - 评估结果对比脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/compare_evaluation_results.py`

## 📋 功能概述

这是多次评估结果对比分析的专用脚本，能够对比不同模型或不同配置下的检测效果，生成详细的对比报告和可视化图表。

### 主要特性
- 📊 多模型性能对比分析
- 📈 丰富的对比可视化图表
- 🎯 关键指标排名和统计
- 📋 详细的对比报告生成
- 💾 支持批量评估结果处理

## 🚀 使用方法

### 基础使用
```bash
# 对比评估目录下的所有结果
python scripts/compare_evaluation_results.py --eval-dir runs/evaluation

# 指定输出目录
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --output-dir runs/comparison/model_comparison
```

### 高级使用
```bash
# 对比指定的评估结果
python scripts/compare_evaluation_results.py \
    --eval-dirs runs/evaluation/model_v1 runs/evaluation/model_v2 runs/evaluation/model_v3 \
    --output-dir runs/comparison/three_models \
    --name model_comparison_v1

# 自定义对比指标
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --metrics mAP@0.5 mAP@0.5:0.95 Precision Recall F1-Score \
    --sort-by mAP@0.5
```

## 📝 参数详解

### 必需参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--eval-dir` | str | 无 | 包含多个评估结果的根目录 | 命令行 |

### 可选参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--eval-dirs` | list | 无 | 指定多个评估结果目录 | 命令行 |
| `--output-dir` | str | `runs/comparison` | 对比结果输出目录 | 命令行 |
| `--name` | str | `comparison` | 对比实验名称 | 命令行 |
| `--project` | str | `runs/comparison` | 项目根目录 | 命令行 |

### 对比配置参数
| 参数 | 类型 | 默认值 | 说明 | 修改位置 |
|------|------|--------|------|----------|
| `--metrics` | list | `['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']` | 对比的指标列表 | 命令行 |
| `--sort-by` | str | `mAP@0.5` | 排序依据指标 | 命令行 |
| `--ascending` | bool | `False` | 是否升序排列 | 命令行 |

## 📂 输入要求

### 评估结果目录结构
```
runs/evaluation/
├── model_v1/
│   ├── evaluation_results.json
│   ├── config.yaml
│   └── *.png
├── model_v2/
│   ├── evaluation_results.json
│   ├── config.yaml
│   └── *.png
└── model_v3/
    ├── evaluation_results.json
    ├── config.yaml
    └── *.png
```

### 必需的输入文件
- **evaluation_results.json**: 评估结果数据文件
- **config.yaml**: 评估配置信息文件

## 📊 输出文件说明

### 对比输出目录结构
```
runs/comparison/comparison/
├── config.yaml                    # 对比实验配置
├── comparison_results.json        # 详细对比数据
├── summary_report.txt             # 文字对比报告
├── metrics_comparison.png         # 指标对比条形图
├── metrics_radar_chart.png        # 雷达图对比
├── ranking_chart.png              # 排名对比图
├── confidence_comparison.png      # 置信度对比图
├── detailed_metrics_table.png     # 详细指标表格
└── performance_trend.png          # 性能趋势图
```

### 主要输出文件解释

#### 1. 对比结果文件 (comparison_results.json)
```json
{
  "comparison_info": {
    "total_models": 3,
    "comparison_date": "2025-05-28 15:30:45",
    "metrics_compared": ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]
  },
  "model_results": {
    "model_v1": {
      "mAP@0.5": 0.8542,
      "mAP@0.5:0.95": 0.6234,
      "Precision": 0.8912,
      "Recall": 0.7834,
      "F1-Score": 0.8339,
      "rank": 1
    },
    "model_v2": {
      "mAP@0.5": 0.8123,
      "mAP@0.5:0.95": 0.5876,
      "Precision": 0.8456,
      "Recall": 0.7923,
      "F1-Score": 0.8181,
      "rank": 2
    }
  },
  "summary_statistics": {
    "best_model": "model_v1",
    "worst_model": "model_v3",
    "avg_mAP_0.5": 0.8234,
    "std_mAP_0.5": 0.0456
  }
}
```

#### 2. 文字对比报告 (summary_report.txt)
```
============================================================
模型性能对比分析报告
============================================================
生成时间: 2025-05-28 15:30:45
对比模型数量: 3
主要排序指标: mAP@0.5

============================================================
整体排名 (按 mAP@0.5 排序)
============================================================
1. model_v1     mAP@0.5: 0.8542  mAP@0.5:0.95: 0.6234
2. model_v2     mAP@0.5: 0.8123  mAP@0.5:0.95: 0.5876  
3. model_v3     mAP@0.5: 0.7891  mAP@0.5:0.95: 0.5234

============================================================
各指标最佳模型
============================================================
- mAP@0.5 最佳: model_v1 (0.8542)
- mAP@0.5:0.95 最佳: model_v1 (0.6234)
- Precision 最佳: model_v1 (0.8912)
- Recall 最佳: model_v2 (0.8123)
- F1-Score 最佳: model_v1 (0.8339)

============================================================
关键发现
============================================================
- model_v1 在大多数指标上表现最佳
- model_v2 在召回率上略胜一筹
- 所有模型的 mAP@0.5 都超过 0.75，表现良好
```

## 🎯 对比指标详解

### 支持的对比指标
- **mAP@0.5**: IoU阈值0.5时的平均精度
- **mAP@0.5:0.95**: IoU阈值0.5到0.95的平均精度
- **Precision**: 精确度，衡量检测准确性
- **Recall**: 召回率，衡量检测完整性
- **F1-Score**: F1分数，综合性能指标
- **Total_TP**: 真正例总数
- **Total_FP**: 假正例总数
- **Total_FN**: 假负例总数

### 排名和统计指标
- **整体排名**: 根据指定指标的排序结果
- **各指标最佳**: 每个指标的最佳表现模型
- **平均值**: 所有模型在各指标上的平均表现
- **标准差**: 模型间性能差异的度量

## 📈 可视化图表说明

### 1. 指标对比条形图 (metrics_comparison.png)
- 并排展示所有模型在各指标上的表现
- 使用不同颜色区分模型
- 显示具体数值标签

### 2. 雷达图对比 (metrics_radar_chart.png)
- 多维度性能对比的雷达图
- 直观展示模型的优势和劣势
- 适合比较模型的综合能力

### 3. 排名对比图 (ranking_chart.png)
- 显示各模型在不同指标上的排名
- 热力图形式展示排名分布
- 便于识别一致性好的模型

### 4. 置信度对比图 (confidence_comparison.png)
- 对比各模型的置信度分布特征
- 展示TP和FP的置信度差异
- 分析模型的判别能力

### 5. 详细指标表格 (detailed_metrics_table.png)
- 以表格形式展示所有详细指标
- 包含排名信息和数值精度
- 便于精确数值比较

## ⚠️ 常见问题和解决方案

### 1. 找不到评估结果文件
```bash
# 问题：指定目录下没有evaluation_results.json文件
# 解决方案：确保先运行evaluate_detection.py生成评估结果
python scripts/evaluate_detection.py --pred-dir ... --gt-dir ... --img-dir ...

# 检查目录结构
ls runs/evaluation/*/evaluation_results.json
```

### 2. 模型数量不足
```bash
# 问题：对比需要至少2个模型的评估结果
# 解决方案：确保有多个不同的评估结果目录
# 每个目录应包含完整的评估结果文件
```

### 3. 指标数据不完整
```bash
# 问题：某些评估结果缺少必要的指标数据
# 解决方案：重新运行评估脚本确保数据完整性
python scripts/evaluate_detection.py --pred-dir ... # 重新评估
```

### 4. 可视化显示异常
```bash
# 问题：图表中文字显示异常或重叠
# 解决方案：脚本已优化字体和布局，通常自动处理
# 如仍有问题，可能是图表内容过多，考虑减少对比模型数量
```

## 🔧 高级使用技巧

### 1. 批量生成对比报告
```bash
# 为不同实验组生成对比报告
for group in group1 group2 group3; do
    python scripts/compare_evaluation_results.py \
        --eval-dir runs/evaluation/${group} \
        --output-dir runs/comparison/${group}_comparison \
        --name ${group}_analysis
done
```

### 2. 自定义指标重要性
```bash
# 根据应用场景调整对比重点
# 注重准确性的场景
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --metrics Precision mAP@0.5:0.95 F1-Score \
    --sort-by Precision

# 注重完整性的场景  
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --metrics Recall mAP@0.5 F1-Score \
    --sort-by Recall
```

### 3. 结合训练配置分析
```bash
# 结合训练配置进行更深入的分析
# 可以查看config.yaml中的训练参数对性能的影响
python scripts/compare_evaluation_results.py \
    --eval-dir runs/evaluation \
    --include-config  # 如果脚本支持此参数
```

## 📊 结果解读指南

### 优秀模型的特征
- **mAP@0.5 > 0.8**: 基础检测性能优秀
- **mAP@0.5:0.95 > 0.6**: 定位精度优秀
- **Precision和Recall均衡**: 避免偏向某一方面
- **置信度分离度高**: TP和FP置信度差异明显

### 模型选择建议
- **生产环境**: 优先选择Precision高的模型
- **研究场景**: 优先选择Recall高的模型
- **平衡应用**: 选择F1-Score最高的模型
- **严格应用**: 选择mAP@0.5:0.95高的模型

### 改进方向识别
- **Precision低**: 考虑提高置信度阈值或改进训练数据质量
- **Recall低**: 考虑降低置信度阈值或增加数据增强
- **mAP@0.5:0.95低**: 需要提高边界框定位精度

## 📈 性能分析方法

### 1. 统计显著性分析
观察标准差值判断模型间差异是否显著：
- 标准差 < 0.02: 模型性能相近
- 标准差 > 0.05: 模型性能差异较大

### 2. 一致性分析
通过排名图分析模型性能的一致性：
- 排名变化小: 模型性能稳定
- 排名变化大: 模型在不同指标上表现不一致

### 3. 综合评分
可以根据业务需求对不同指标赋予权重：
```
综合评分 = 0.4×mAP@0.5 + 0.3×Precision + 0.2×Recall + 0.1×F1-Score
```

## 🔗 相关文档

- [evaluate_detection.md](evaluate_detection.md) - 单模型评估分析
- [small_target_train.md](small_target_train.md) - 模型训练参数调优
- [generate_charts.md](generate_charts.md) - 训练过程可视化