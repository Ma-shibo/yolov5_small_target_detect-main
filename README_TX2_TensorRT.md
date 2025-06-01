# TX2/NX Real-time YOLOv5 Detection with TensorRT

## 功能特性

- 🚀 TensorRT加速推理，在TX2/NX上可获得2-5倍性能提升
- 📷 GStreamer硬件加速相机管道
- 🎥 硬件编码视频录制
- 📊 实时性能监控
- 🔧 自动模型导出和优化

## 安装要求

### 基础依赖
```bash
# 安装Python依赖
pip install torch torchvision opencv-python numpy

# 安装GStreamer开发包 (Ubuntu/Debian)
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

# 安装Python GStreamer绑定
pip install PyGObject
```

### TensorRT安装 (可选但推荐)
```bash
# TX2/NX通常已预装TensorRT，如需安装：
# 1. 从NVIDIA下载TensorRT
# 2. 安装TensorRT Python包
pip install pycuda
```

## 使用方法

### 1. 基础实时检测 (PyTorch)
```bash
python tx2_realtime_detect.py --weights yolov5s.pt --view-img
```

### 2. TensorRT加速检测
```bash
# 自动导出并使用TensorRT
python tx2_realtime_detect.py --weights yolov5s.pt --trt --export-trt

# 使用已导出的TensorRT引擎
python tx2_realtime_detect.py --weights yolov5s.engine --trt
```

### 3. 高级选项
```bash
# 高性能设置 + 录制视频
python tx2_realtime_detect.py \
    --weights yolov5s.pt \
    --trt --export-trt \
    --half \
    --imgsz 416 \
    --conf-thres 0.4 \
    --camera-width 1280 \
    --camera-height 720 \
    --save-video \
    --trt-workspace 2048

# 无显示模式（适合远程部署）
python tx2_realtime_detect.py \
    --weights yolov5s.pt \
    --trt \
    --no-display \
    --save-images
```

## 参数说明

### 模型参数
- `--weights`: 模型文件路径 (.pt或.engine)
- `--device`: CUDA设备ID (默认自动选择)
- `--conf-thres`: 置信度阈值 (0.0-1.0)
- `--iou-thres`: NMS IoU阈值 (0.0-1.0)
- `--imgsz`: 推理图像尺寸 (建议: 320/416/640)
- `--half`: 使用FP16半精度 (提升性能)

### TensorRT参数
- `--trt`: 启用TensorRT推理
- `--export-trt`: 导出TensorRT引擎
- `--trt-workspace`: TensorRT工作空间大小(MB)

### 相机参数
- `--camera-width/height`: 相机分辨率
- `--camera-fps`: 相机帧率

### 输出参数
- `--save-video`: 保存检测视频
- `--save-images`: 保存检测图像
- `--output-dir`: 输出目录
- `--no-display`: 禁用显示窗口

## 性能优化建议

### TX2优化设置
```bash
# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 推荐配置
python tx2_realtime_detect.py \
    --weights yolov5s.pt \
    --trt --export-trt \
    --half \
    --imgsz 416 \
    --camera-width 1280 \
    --camera-height 720 \
    --camera-fps 30
```

### NX优化设置
```bash
# NX可以使用更高分辨率和更大模型
python tx2_realtime_detect.py \
    --weights yolov5m.pt \
    --trt --export-trt \
    --half \
    --imgsz 640 \
    --camera-width 1920 \
    --camera-height 1080 \
    --trt-workspace 4096
```

## 性能对比

| 设备 | 模型 | 后端 | 分辨率 | FPS |
|------|------|------|--------|-----|
| TX2 | YOLOv5s | PyTorch | 640×640 | ~8 FPS |
| TX2 | YOLOv5s | TensorRT | 640×640 | ~15 FPS |
| TX2 | YOLOv5s | TensorRT | 416×416 | ~25 FPS |
| NX | YOLOv5s | TensorRT | 640×640 | ~30 FPS |
| NX | YOLOv5m | TensorRT | 640×640 | ~20 FPS |

## 故障排除

### 常见问题

1. **TensorRT导出失败**
   ```bash
   # 检查CUDA和TensorRT版本兼容性
   python -c "import tensorrt as trt; print(trt.__version__)"
   ```

2. **相机无法打开**
   ```bash
   # 检查相机设备
   ls /dev/video*
   # 测试GStreamer管道
   gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
   ```

3. **内存不足**
   ```bash
   # 减少工作空间大小
   --trt-workspace 1024
   # 使用更小的输入尺寸
   --imgsz 320
   ```

4. **帧率低**
   ```bash
   # 启用最大性能模式
   sudo nvpmodel -m 0 && sudo jetson_clocks
   # 使用TensorRT + FP16
   --trt --half
   # 降低相机分辨率
   --camera-width 1280 --camera-height 720
   ```

## 自定义模型

如果使用自定义训练的模型：

1. 确保模型与YOLOv5格式兼容
2. 更新类别名称（如果需要）
3. 调整置信度阈值
4. 重新导出TensorRT引擎

```bash
python tx2_realtime_detect.py \
    --weights custom_model.pt \
    --trt --export-trt \
    --conf-thres 0.3
```