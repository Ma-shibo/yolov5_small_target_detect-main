# gui_test.py - 图形界面测试工具

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/gui_test.py`

## 📋 功能概述

这是一个功能完整的图形界面测试工具，提供直观的YOLOv5小目标检测模型测试界面，支持参数配置、实时日志显示和TensorRT加速。

### 主要特性
- 🖥️ 直观的图形用户界面
- 🔧 完整的参数配置选项
- 📊 实时训练/测试日志显示
- ⚡ TensorRT引擎生成和加速
- 🎯 支持图片、视频和批量检测
- 💾 配置保存和加载功能

## 🚀 使用方法

### 启动GUI
```bash
# 启动图形界面
python scripts/gui_test.py

# 在后台运行
nohup python scripts/gui_test.py &
```

### 界面操作流程
1. **选择权重文件**: 点击"浏览"选择模型权重(.pt或.engine文件)
2. **选择测试源**: 根据测试模式选择图片目录或视频目录
3. **配置参数**: 调整置信度、IoU阈值等检测参数
4. **启动测试**: 点击"开始测试"按钮
5. **查看结果**: 在日志区域查看实时输出和结果

## 📝 界面组件详解

### 文件选择区域
| 组件 | 功能 | 支持格式 | 说明 |
|------|------|----------|------|
| 权重文件 | 选择模型权重 | `.pt`, `.engine` | 支持PyTorch权重和TensorRT引擎 |
| 测试源 | 选择测试数据 | 目录路径 | 根据测试模式自动切换选择类型 |
| 输出目录 | 设置结果保存路径 | 目录路径 | 默认为`runs/detect/gui_test` |

### 测试模式选择
| 模式 | 说明 | 适用场景 | 输出结果 |
|------|------|----------|----------|
| 视频检测 | 检测视频文件中的目标 | 单个或多个视频文件 | 标注视频、标签文件 |
| 图片检测 | 检测图片中的目标 | 图片目录 | 标注图片、标签文件 |
| 批量检测 | 批量处理多种格式文件 | 混合格式数据 | 按格式分类输出 |

### 基本参数配置
| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| 置信度阈值 | 滑块 | 0.01-1.0 | 0.25 | 检测置信度门槛 |
| IOU阈值 | 滑块 | 0.01-1.0 | 0.45 | NMS去重阈值 |
| 图像尺寸 | 下拉框 | 320,416,512,640,832,1280 | 640 | 输入图像尺寸 |
| 设备 | 下拉框 | cpu,0,1,0,1 | 0 | 推理设备选择 |
| 最大检测数 | 数值框 | 1-10000 | 1000 | 单张图片最大检测目标数 |

### 功能选项开关
| 选项 | 默认值 | 说明 | 输出影响 |
|------|--------|------|----------|
| 保存标签文件 | ✅ | 保存YOLO格式标签 | 生成.txt标签文件 |
| 保存置信度 | ✅ | 在标签中包含置信度 | 标签文件包含confidence值 |
| 保存裁剪图像 | ✅ | 保存检测到的目标裁剪图 | 生成crops子目录 |
| 显示结果 | ❌ | 实时显示检测结果 | 弹窗显示标注图像 |
| 半精度推理 | ✅ | 使用FP16加速推理 | 提升速度，略微降低精度 |
| 数据增强 | ❌ | 推理时使用TTA | 提高准确性，降低速度 |

### 高级设置
| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| 最大测试帧数 | 数值框 | 1-10000 | 300 | 视频检测最大处理帧数 |
| 视频步幅 | 数值框 | 1-10 | 1 | 视频帧采样间隔 |
| 线条粗细 | 数值框 | 1-10 | 3 | 标注框线条粗细 |
| 启用时间范围 | 复选框 | - | ❌ | 仅处理视频指定时间段 |
| 起始时间 | 文本框 | HH:MM:SS | 00:00:00 | 视频处理起始时间 |
| 结束时间 | 文本框 | HH:MM:SS | 00:01:00 | 视频处理结束时间 |

### TensorRT加速设置
| 功能 | 说明 | 使用条件 | 性能提升 |
|------|------|----------|----------|
| 启用TensorRT加速 | 使用TensorRT引擎推理 | 需要NVIDIA GPU | 2-5倍速度提升 |
| 生成TensorRT引擎 | 从PyTorch权重生成引擎 | 首次使用需要 | 一次性生成，后续复用 |

## 📊 输出结果说明

### 检测输出目录结构
```
runs/detect/gui_test/exp/
├── labels/                     # YOLO格式标签文件
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── crops/                      # 裁剪的目标图像
│   └── target/
│       ├── image1_target_1.jpg
│       ├── image1_target_2.jpg
│       └── ...
├── image1.jpg                  # 标注后的图像
├── image2.jpg
├── video1.mp4                  # 标注后的视频
└── results.txt                 # 检测统计信息
```

### 标签文件格式 (labels/*.txt)
```
# 格式: class_id x_center y_center width height [confidence]
0 0.5 0.3 0.1 0.2 0.85
0 0.7 0.6 0.15 0.25 0.92
```

### 日志信息解读
```
[14:30:15] 🚀 开始YOLOv5测试
[14:30:15] 📁 权重文件: best.pt
[14:30:15] 📁 测试源: /path/to/videos/
[14:30:15] 📁 测试模式: video
[14:30:16] ⚡ 模型加载完成
[14:30:16] 🎯 开始处理: video1.mp4
[14:30:18] ✅ 检测完成: 发现 15 个目标
[14:30:18] 💾 结果保存至: runs/detect/gui_test/exp
```

## 🔧 配置文件功能

### 保存配置 (config.json)
```json
{
  "weights_path": "runs/small_target_train/exp/weights/best.pt",
  "source_path": "/home/lkx/Videos/test_videos/",
  "output_path": "runs/detect/gui_test",
  "test_mode": "video",
  "conf_thres": 0.25,
  "iou_thres": 0.45,
  "max_det": 1000,
  "imgsz": 640,
  "device": "0",
  "save_txt": true,
  "save_conf": true,
  "save_crop": true,
  "view_img": false,
  "half_precision": true,
  "use_tensorrt": false,
  "augment": false,
  "max_frames": 300,
  "vid_stride": 1,
  "line_thickness": 3
}
```

### 加载配置
点击"加载配置"按钮可以导入之前保存的配置文件，所有参数将自动设置为保存的值。

## ⚡ TensorRT加速使用

### 生成TensorRT引擎
1. 确保已安装TensorRT和相关依赖
2. 在GUI中勾选"启用TensorRT加速"
3. 点击"生成TensorRT引擎"按钮
4. 等待引擎生成完成（首次较慢）
5. 生成的.engine文件可重复使用

### TensorRT要求
- NVIDIA GPU (支持CUDA)
- TensorRT 8.0+
- CUDA 11.0+
- cuDNN 8.0+

### 性能对比
| 推理方式 | 相对速度 | 精度影响 | 内存使用 |
|----------|----------|----------|----------|
| PyTorch FP32 | 1x | 基准 | 高 |
| PyTorch FP16 | 1.5-2x | 微小下降 | 中等 |
| TensorRT FP32 | 2-3x | 无影响 | 中等 |
| TensorRT FP16 | 3-5x | 微小下降 | 低 |

## ⚠️ 常见问题和解决方案

### 1. GUI无法启动
```bash
# 问题：缺少tkinter库
# 解决方案：
sudo apt-get install python3-tk  # Ubuntu/Debian
# 或
conda install tk  # Conda环境
```

### 2. 中文字体显示问题
```bash
# 问题：GUI中中文显示为方框
# 解决方案：脚本已自动使用系统默认字体，通常无需额外配置
# 如仍有问题，检查系统中文字体安装
fc-list :lang=zh
```

### 3. TensorRT引擎生成失败
```bash
# 问题：TensorRT环境未正确配置
# 解决方案：
# 1. 检查CUDA版本兼容性
nvidia-smi
# 2. 检查TensorRT安装
python -c "import tensorrt; print(tensorrt.__version__)"
# 3. 重新安装TensorRT
```

### 4. 检测结果不理想
```bash
# 问题：检测精度不够或误检率高
# 解决方案：
# 1. 调整置信度阈值 (提高 = 减少误检，降低 = 减少漏检)
# 2. 调整IoU阈值 (提高 = 减少重复检测)
# 3. 尝试不同的图像尺寸
# 4. 检查模型是否适合当前场景
```

### 5. 处理速度慢
```bash
# 问题：检测速度过慢
# 解决方案：
# 1. 启用半精度推理
# 2. 使用TensorRT加速
# 3. 降低图像尺寸
# 4. 增加视频步幅(仅视频检测)
```

## 📈 性能优化建议

### 硬件优化
- 使用NVIDIA GPU进行推理
- 确保足够的GPU显存(推荐8GB+)
- 使用SSD存储以提高I/O速度

### 软件优化
- 启用TensorRT加速
- 使用半精度推理(FP16)
- 合理设置批处理大小
- 关闭不必要的可视化选项

### 参数调优
- 根据具体场景调整置信度阈值
- 选择合适的图像尺寸(平衡速度与精度)
- 对于实时应用，可适当降低最大检测数

## 🔗 相关脚本集成

### 与其他脚本的配合使用
```bash
# 1. 使用GUI测试后，评估结果
python scripts/evaluate_detection.py \
    --pred-dir runs/detect/gui_test/exp/labels \
    --gt-dir /path/to/ground_truth/labels \
    --img-dir /path/to/images

# 2. 批量测试多个视频
python scripts/test_videos.py \
    --weights runs/small_target_train/exp/weights/best.pt \
    --source /path/to/videos/

# 3. 生成训练图表
python scripts/generate_charts.py \
    --results runs/small_target_train/exp/results.csv
```

## 🎯 使用技巧

### 快速测试流程
1. 保存常用配置到config.json
2. 使用"加载配置"快速设置参数
3. 先用小批量数据测试参数效果
4. 确认参数后进行大批量处理

### 调试技巧
1. 启用"显示结果"查看实时检测效果
2. 查看日志了解处理进度和错误信息
3. 检查crops目录确认检测质量
4. 对比标注图像与原图判断效果

## 🔗 相关文档

- [test_videos.md](test_videos.md) - 批量视频测试
- [evaluate_detection.md](evaluate_detection.md) - 结果评估分析
- [small_target_train.md](small_target_train.md) - 模型训练