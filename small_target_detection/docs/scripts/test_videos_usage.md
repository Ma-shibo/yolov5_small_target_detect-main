# 视频检测测试脚本使用文档

**对应脚本文件：** `/small_target_detection/scripts/test_videos.py`

## 目录
- [脚本概述](#脚本概述)
- [功能特性](#功能特性)
- [参数说明](#参数说明)
- [使用方法](#使用方法)
- [输出文件说明](#输出文件说明)
- [ROI设置详解](#roi设置详解)
- [性能优化](#性能优化)
- [结果分析](#结果分析)
- [注意事项](#注意事项)
- [常见问题](#常见问题)

## 脚本概述

`test_videos.py` 是YOLOv5小目标检测的视频/图片批量测试脚本。支持对视频文件或图片文件夹进行检测，提供丰富的可视化输出、统计分析和性能监控功能。

## 功能特性

- 🎬 支持视频文件和图片文件夹检测
- ⏰ 支持时间范围检测（仅视频）
- 🎯 支持ROI区域检测
- 📊 详细的统计信息和性能分析
- 💾 多种输出格式（视频、图片、标签、裁剪）
- 🚀 半精度推理优化
- 📈 实时进度监控

## 参数说明

### 基础参数

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--weights` | str | `'runs/small_target_train/exp7/weights/best.pt'` | 模型权重文件路径 | `.pt` 文件路径 |
| `--source` | str | `'/home/lkx/Videos/2025_CUADC_FORWARD/'` | 视频源路径 | 视频文件或图片文件夹 |
| `--conf-thres` | float | `0.15` | 置信度阈值 | `0.0-1.0` |
| `--iou-thres` | float | `0.45` | NMS IOU阈值 | `0.0-1.0` |
| `--max-det` | int | `1000` | 最大检测数量 | 正整数 |
| `--imgsz` | int/list | `[640]` | 推理图像尺寸 | 正整数或`[height, width]` |
| `--device` | str | `'0'` | 计算设备 | `'0'`, `'1'`, `'cpu'`, `'0,1'` |

### 输出控制参数

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--save-txt` | bool | `False` | 保存YOLO格式标签 | `True/False` |
| `--save-conf` | bool | `False` | 在标签中保存置信度 | `True/False` |
| `--save-crop` | bool | `False` | 保存检测目标裁剪图片 | `True/False` |
| `--save-as-images` | bool | `False` | 保存为单独图片而非视频 | `True/False` |
| `--nosave` | bool | `False` | 不保存检测结果 | `True/False` |
| `--view-img` | bool | `False` | 实时显示检测结果 | `True/False` |

### 时间控制参数（仅视频）

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--start-time` | str | `None` | 开始时间 | `'HH:MM:SS'` 或 `'MM:SS'` |
| `--end-time` | str | `None` | 结束时间 | `'HH:MM:SS'` 或 `'MM:SS'` |
| `--max-frames` | int | `3000` | 最大处理帧数 | 正整数 |
| `--vid-stride` | int | `1` | 视频帧步长 | 正整数 |

### ROI参数

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--roi-params` | float | `None` | ROI区域参数 | `x y width height` |
| `--roi-mode` | str | `'percentage'` | ROI模式 | `'percentage'`, `'absolute'` |

### 性能优化参数

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--half` | bool | `False` | 使用FP16半精度推理 | `True/False` |
| `--augment` | bool | `False` | 启用测试时增强 | `True/False` |
| `--agnostic-nms` | bool | `False` | 类别无关NMS | `True/False` |

## 使用方法

### 1. 基本视频检测

```bash
# 检测单个视频文件
python test_videos.py \
    --weights runs/small_target_train/exp7/weights/best.pt \
    --source /path/to/video.mp4 \
    --conf-thres 0.15 \
    --save-txt \
    --save-crop
```

### 2. 图片文件夹检测

```bash
# 检测图片文件夹
python test_videos.py \
    --weights runs/small_target_train/exp7/weights/best.pt \
    --source /path/to/images/ \
    --save-as-images \
    --save-txt \
    --save-crop
```

### 3. 时间范围检测

```bash
# 检测视频的指定时间段
python test_videos.py \
    --weights runs/small_target_train/exp7/weights/best.pt \
    --source /path/to/video.mp4 \
    --start-time 00:01:30 \
    --end-time 00:05:00 \
    --max-frames 1000
```

### 4. ROI区域检测

```bash
# 检测指定ROI区域（百分比模式）
python test_videos.py \
    --weights runs/small_target_train/exp7/weights/best.pt \
    --source /path/to/video.mp4 \
    --roi-params 20 30 60 40 \
    --roi-mode percentage
```

### 5. 高性能检测

```bash
# 启用性能优化
python test_videos.py \
    --weights runs/small_target_train/exp7/weights/best.pt \
    --source /path/to/video.mp4 \
    --half \
    --vid-stride 2 \
    --max-frames 5000
```

## 输出文件说明

### 1. 检测结果视频/图片
- **位置**: `runs/detect/single_video_detailed_test/`
- **格式**: 
  - 视频: `{video_name}_detected.mp4`
  - 图片: `images/{image_name}_detected.jpg`
- **内容**: 带有检测框和标签的可视化结果

### 2. YOLO标签文件（`--save-txt`）
- **位置**: `runs/detect/single_video_detailed_test/labels/`
- **格式**: 
  - 视频: `{video_name}_frame_{frame:06d}.txt`
  - 图片: `{image_name}.txt`
- **内容**: YOLO格式标注
```
class_id x_center y_center width height confidence
0 0.5 0.3 0.1 0.08 0.95
```

### 3. 裁剪图片（`--save-crop`）
- **位置**: `runs/detect/single_video_detailed_test/crops/{class_name}/`
- **格式**: `{source_name}_{confidence:.3f}.jpg`
- **内容**: 检测目标的裁剪图片

### 4. 统计数据
- **详细统计**: `statistics/detailed_results.json`
- **帧统计**: `statistics/frame_statistics.csv`

#### detailed_results.json 结构
```json
{
    "video_info": {
        "path": "/path/to/video.mp4",
        "name": "video.mp4",
        "total_frames": 1500,
        "weights_used": "best.pt"
    },
    "detection_summary": {
        "person": {
            "count": 150,
            "total_conf": 135.5,
            "avg_conf": 0.903
        }
    },
    "frame_statistics": {
        "0": {
            "detections_count": 2,
            "inference_time": 15.2,
            "detections": [...],
            "time_seconds": 0.0
        }
    },
    "performance": {
        "avg_preprocess_time": 2.1,
        "avg_inference_time": 15.3,
        "avg_nms_time": 1.8,
        "total_time": 45.2,
        "fps": 33.2
    }
}
```

## ROI设置详解

### 百分比模式（推荐）
```bash
--roi-params 20 30 60 40 --roi-mode percentage
```
- `20`: X起始位置（图像宽度的20%）
- `30`: Y起始位置（图像高度的30%）
- `60`: ROI宽度（图像宽度的60%）
- `40`: ROI高度（图像高度的40%）

### 绝对像素模式
```bash
--roi-params 200 300 800 600 --roi-mode absolute
```
- `200`: X起始像素
- `300`: Y起始像素
- `800`: ROI宽度像素
- `600`: ROI高度像素

## 性能优化

### 1. 硬件优化
- **GPU加速**: `--device 0`
- **半精度推理**: `--half`（提速2倍，轻微精度损失）
- **多GPU**: `--device 0,1`

### 2. 处理优化
- **跳帧处理**: `--vid-stride 2`（每2帧处理1帧）
- **限制帧数**: `--max-frames 1000`
- **减少输出**: `--nosave`（仅统计，不保存结果）

### 3. 模型优化
- **较小模型**: 使用yolov5s而非yolov5x
- **图像尺寸**: `--imgsz 416`（较小尺寸提速）

## 结果分析

### 1. 检测效果评估
```python
# 分析检测统计
import json
with open('statistics/detailed_results.json', 'r') as f:
    results = json.load(f)

total_detections = sum(stats['count'] for stats in results['detection_summary'].values())
avg_confidence = sum(stats['avg_conf'] * stats['count'] for stats in results['detection_summary'].values()) / total_detections
```

### 2. 性能分析
- **推理速度**: 查看 `performance.fps`
- **检测密度**: `total_detections / total_frames`
- **置信度分布**: 各类别的 `avg_conf`

### 3. 时序分析
```python
import pandas as pd
df = pd.read_csv('statistics/frame_statistics.csv')
print(f"平均每帧检测数: {df['detections'].mean():.2f}")
print(f"最大单帧检测数: {df['detections'].max()}")
```

## 注意事项

### ⚠️ 重要提醒

1. **内存使用**: 大视频文件可能消耗大量内存，建议设置 `--max-frames`
2. **存储空间**: 启用 `--save-crop` 可能产生大量文件
3. **处理时间**: 高分辨率视频处理时间较长
4. **权重文件**: 确保权重文件与当前模型配置匹配

### 🔧 修改位置

要修改脚本的默认行为，可以编辑以下位置：

```python
# 文件: test_videos.py
# 修改默认参数（第25-35行附近）
def run_single_video_test(
    weights='runs/small_target_train/exp7/weights/best.pt',
    source='/home/lkx/Videos/2025_CUADC_FORWARD/',
    conf_thres=0.15,  # 置信度阈值
    max_frames=3000,  # 最大帧数
    # ...
):
```

## 常见问题

### Q1: 视频检测速度太慢怎么办？
**A**: 使用以下优化策略：
```bash
python test_videos.py \
    --weights your_model.pt \
    --source video.mp4 \
    --half \
    --vid-stride 3 \
    --max-frames 1000 \
    --imgsz 416
```

### Q2: 如何只检测视频的特定区域？
**A**: 使用ROI参数：
```bash
python test_videos.py \
    --source video.mp4 \
    --roi-params 25 25 50 50 \
    --roi-mode percentage
```

### Q3: 检测结果置信度普遍偏低怎么办？
**A**: 
1. 降低置信度阈值: `--conf-thres 0.1`
2. 检查模型是否适合当前场景
3. 考虑重新训练或微调模型

### Q4: 如何批量处理多个视频？
**A**: 创建批处理脚本：
```bash
#!/bin/bash
for video in /path/to/videos/*.mp4; do
    python test_videos.py \
        --source "$video" \
        --weights best.pt \
        --save-txt \
        --save-crop
done
```

### Q5: 内存不足怎么办？
**A**: 
1. 减少最大帧数: `--max-frames 500`
2. 使用较小图像尺寸: `--imgsz 416`
3. 关闭不必要的保存选项
4. 增加跳帧: `--vid-stride 5`

### Q6: 如何提取特定时间段的检测结果？
**A**: 
```bash
python test_videos.py \
    --source video.mp4 \
    --start-time 00:02:30 \
    --end-time 00:05:15 \
    --save-as-images
```

---

**最后更新**: 2025年5月28日
**维护者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)