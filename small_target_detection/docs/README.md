# YOLOv5 小目标检测项目文档

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日

## 📁 项目结构

```
small_target_detection/
├── scripts/               # 核心脚本文件
│   ├── small_target_train.py         # 训练脚本
│   ├── evaluate_detection.py         # 检测结果评估脚本
│   ├── compare_evaluation_results.py # 评估结果对比分析脚本
│   ├── generate_charts.py            # 训练图表生成脚本
│   ├── gui_test.py                   # 图形界面测试工具
│   ├── start_training.py             # 训练启动脚本
│   ├── training_monitor.py           # 训练监控脚本
│   ├── test_videos.py                # 视频测试脚本
│   └── update_labels_class.py        # 标签类别更新脚本
├── docs/                  # 文档目录
│   ├── README.md                     # 本文档
│   ├── scripts/                      # 各脚本详细文档
│   │   ├── small_target_train.md     # 训练脚本文档
│   │   ├── evaluate_detection.md     # 评估脚本文档
│   │   ├── compare_evaluation_results.md # 对比分析文档
│   │   ├── generate_charts.md        # 图表生成文档
│   │   ├── gui_test.md               # GUI测试工具文档
│   │   ├── start_training.md         # 训练启动文档
│   │   ├── training_monitor.md       # 训练监控文档
│   │   ├── test_videos.md            # 视频测试文档
│   │   └── update_labels_class.md    # 标签更新文档
│   └── YOLOv5小目标检测完整使用文档.md
├── configs/               # 配置文件
├── logs/                  # 日志文件
└── weights/               # 权重文件
```

## 🚀 快速开始

### 1. 训练模型
```bash
# 使用默认配置训练
python scripts/small_target_train.py

# 自定义配置训练
python scripts/small_target_train.py --data dataset.yaml --epochs 200 --batch-size 16
```

### 2. 评估模型
```bash
# 评估检测结果
python scripts/evaluate_detection.py --pred-dir runs/detect/exp/labels --gt-dir data/labels --img-dir data/images
```

### 3. 图形界面测试
```bash
# 启动GUI测试工具
python scripts/gui_test.py
```

## 📋 脚本功能概览

| 脚本文件 | 主要功能 | 输入 | 输出 | 文档链接 |
|---------|---------|------|------|---------|
| `small_target_train.py` | 小目标检测模型训练 | 数据集、配置文件 | 训练权重、日志 | [详细文档](scripts/small_target_train.md) |
| `evaluate_detection.py` | 检测结果评估分析 | 预测标签、真实标签、图片 | 评估报告、可视化图表 | [详细文档](scripts/evaluate_detection.md) |
| `compare_evaluation_results.py` | 多次评估结果对比 | 多个评估结果目录 | 对比分析报告 | [详细文档](scripts/compare_evaluation_results.md) |
| `generate_charts.py` | 训练结果图表生成 | results.csv文件 | 训练曲线图表 | [详细文档](scripts/generate_charts.md) |
| `gui_test.py` | 图形界面测试工具 | 权重、测试数据 | 检测结果 | [详细文档](scripts/gui_test.md) |
| `start_training.py` | 训练任务启动器 | 训练配置 | 训练进程 | [详细文档](scripts/start_training.md) |
| `training_monitor.py` | 训练过程监控 | 训练日志 | 监控报告 | [详细文档](scripts/training_monitor.md) |
| `test_videos.py` | 视频批量测试 | 视频文件、权重 | 检测结果视频 | [详细文档](scripts/test_videos.md) |
| `update_labels_class.py` | 标签类别更新 | 标签文件 | 更新后标签 | [详细文档](scripts/update_labels_class.md) |

## 🔧 环境配置

### 依赖包安装
```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install opencv-python matplotlib pandas numpy
pip install albumentations seaborn tqdm pyyaml

# GUI依赖
pip install tkinter  # 通常已内置

# 可选依赖
pip install tensorboard  # 训练监控
pip install wandb       # 实验跟踪
```

### GPU配置
```bash
# 检查CUDA版本
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 典型工作流程

### 完整训练评估流程
```bash
# 1. 启动训练
python scripts/small_target_train.py --data dataset.yaml --epochs 200

# 2. 监控训练过程
python scripts/training_monitor.py --log-dir runs/small_target_train/exp

# 3. 生成训练图表
python scripts/generate_charts.py --results runs/small_target_train/exp/results.csv

# 4. 使用GUI测试模型
python scripts/gui_test.py

# 5. 评估检测结果
python scripts/evaluate_detection.py --pred-dir runs/detect/exp/labels

# 6. 对比多次评估结果
python scripts/compare_evaluation_results.py --eval-dir runs/evaluation
```

### 批量视频测试流程
```bash
# 1. 批量测试视频
python scripts/test_videos.py --weights best.pt --source /path/to/videos/

# 2. 评估测试结果
python scripts/evaluate_detection.py --pred-dir runs/detect/exp/labels
```

## 🎯 最佳实践

### 训练建议
- 使用`--augment-ratio 0.3-0.5`进行适度数据增强
- 小目标检测推荐`--imgsz 640`或更高
- 使用`--cos-lr`余弦学习率调度
- 启用`--image-weights`加权采样

### 评估建议
- 使用多个IoU阈值评估：`--iou-thresholds 0.5 0.6 0.7 0.8 0.9`
- 生成样本图像对比：自动生成sample_images目录
- 对比多次实验结果找出最佳配置

### 部署建议
- 使用GUI工具快速验证模型效果
- 批量测试验证模型泛化性能
- 导出TensorRT引擎加速推理

## 🔍 故障排除

### 常见问题
1. **CUDA内存不足**：降低batch-size或使用--cache ram
2. **标签格式错误**：使用update_labels_class.py修正
3. **训练不收敛**：调整学习率或数据增强强度
4. **GUI显示异常**：检查系统字体配置

### 调试工具
- 使用`training_monitor.py`实时监控训练
- 使用`evaluate_detection.py`生成详细分析报告
- 使用`compare_evaluation_results.py`对比不同配置效果

## 📝 更新日志

### v1.0.0 (2025-05-28)
- ✅ 完善所有脚本文档
- ✅ 增加详细参数说明
- ✅ 添加输出文件解释
- ✅ 完善故障排除指南
- ✅ 更新作者和时间信息

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进文档和代码质量。

## 📧 联系方式

如有问题请提交Issue或联系项目维护者：2022级-飞行器设计与工程-雒凯星(多旋翼项目组)