# YOLOv5 ROS Package

一个基于YOLOv5的ROS包，支持实时目标检测和TensorRT加速，专为NVIDIA TX2/NX平台优化。

## 功能特性

- 🚀 **实时目标检测**: 基于YOLOv5的高精度目标检测
- ⚡ **TensorRT加速**: 支持TensorRT推理加速，在TX2/NX上可获得2-5倍性能提升
- 📷 **硬件加速相机**: 使用GStreamer硬件加速的相机驱动
- 🔄 **ROS集成**: 完整的ROS话题发布和订阅机制
- ⏱️ **时间同步**: 统一的时间戳管理，便于多传感器融合
- 📊 **性能监控**: 实时FPS和推理时间统计

## ROS话题

### 发布的话题

- `/yolov5/detections` (`yolov5_ros/DetectionArray`): 检测结果数组
- `/yolov5/detection_image` (`sensor_msgs/Image`): 带检测框的可视化图像
- `/camera/image_raw` (`sensor_msgs/Image`): 原始相机图像
- `/camera/camera_info` (`sensor_msgs/CameraInfo`): 相机标定信息

### 订阅的话题

- `/camera/image_raw` (`sensor_msgs/Image`): 输入图像数据

### 自定义消息类型

#### BoundingBox.msg
```
float64 x      # 左上角x坐标
float64 y      # 左上角y坐标  
float64 width  # 边界框宽度
float64 height # 边界框高度
```

#### Detection.msg
```
std_msgs/Header header
string class_name     # 检测对象类别名称
int32 class_id       # 检测对象类别ID
float64 confidence   # 检测置信度
BoundingBox bbox     # 边界框坐标
geometry_msgs/Point center  # 检测中心点
```

#### DetectionArray.msg
```
std_msgs/Header header
sensor_msgs/Image image        # 原始图像
Detection[] detections         # 检测结果数组
int32 detection_count         # 检测数量
float64 inference_time        # 推理时间(秒)
```

## 安装和编译

### 1. 依赖安装

```bash
# 基础ROS依赖
sudo apt-get install ros-melodic-cv-bridge ros-melodic-image-transport

# Python依赖
pip3 install torch torchvision opencv-python numpy

# GStreamer支持(TX2/NX)
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
pip3 install PyGObject

# TensorRT支持(可选)
pip3 install pycuda
```

### 2. 编译ROS包

```bash
# 进入catkin工作空间
cd ~/catkin_ws/src

# 克隆或复制yolov5_ros包到src目录
cp -r /path/to/yolov5/yolov5_ros .

# 编译
cd ~/catkin_ws
catkin_make

# 更新环境变量
source devel/setup.bash
```

## 使用方法

### 1. 基础实时检测

```bash
# 启动基础检测
roslaunch yolov5_ros yolov5_detection.launch

# 自定义参数
roslaunch yolov5_ros yolov5_detection.launch \
    weights:=/path/to/your/model.pt \
    conf_thres:=0.4 \
    camera_width:=1280 \
    camera_height:=720
```

### 2. TensorRT加速检测(推荐用于TX2/NX)

```bash
# 启动TensorRT加速检测
roslaunch yolov5_ros yolov5_tensorrt.launch

# 自定义TensorRT设置
roslaunch yolov5_ros yolov5_tensorrt.launch \
    weights:=/path/to/your/model.pt \
    use_trt:=true \
    half:=true \
    imgsz:=416
```

### 3. 查看检测结果

```bash
# 查看检测话题
rostopic echo /yolov5/detections

# 查看图像话题
rosrun image_view image_view image:=/yolov5/detection_image

# 查看话题列表
rostopic list | grep yolov5
```

### 4. 性能监控

```bash
# 查看话题频率
rostopic hz /yolov5/detections

# 查看消息信息
rostopic info /yolov5/detections

# 实时日志查看
rosnode info yolov5_detection
```

## 参数配置

### 检测参数

- `weights`: 模型文件路径 (.pt或.engine)
- `device`: CUDA设备ID (默认自动选择)
- `conf_thres`: 置信度阈值 (0.0-1.0)
- `iou_thres`: NMS IoU阈值 (0.0-1.0)
- `imgsz`: 推理图像尺寸
- `half`: 使用FP16半精度推理
- `use_trt`: 启用TensorRT加速

### 相机参数

- `camera_width/height`: 相机分辨率
- `camera_fps`: 相机帧率
- `sensor_id`: 相机传感器ID

### 话题参数

- `image_topic`: 输入图像话题名
- `detection_topic`: 检测结果话题名
- `visualize`: 启用可视化
- `publish_image`: 发布检测图像

## TX2/NX优化建议

### 1. 设置最大性能模式

```bash
# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2. 推荐配置

**TX2高性能配置:**
```bash
roslaunch yolov5_ros yolov5_tensorrt.launch \
    weights:=yolov5s.pt \
    use_trt:=true \
    half:=true \
    imgsz:=416 \
    camera_width:=1280 \
    camera_height:=720 \
    conf_thres:=0.4
```

**NX高性能配置:**
```bash
roslaunch yolov5_ros yolov5_tensorrt.launch \
    weights:=yolov5m.pt \
    use_trt:=true \
    half:=true \
    imgsz:=640 \
    camera_width:=1920 \
    camera_height:=1080
```

## 性能基准

| 平台 | 模型 | 后端 | 分辨率 | FPS | 推理时间 |
|------|------|------|--------|-----|----------|
| TX2 | YOLOv5s | PyTorch | 640×640 | ~8 | ~125ms |
| TX2 | YOLOv5s | TensorRT | 640×640 | ~15 | ~67ms |
| TX2 | YOLOv5s | TensorRT | 416×416 | ~25 | ~40ms |
| NX | YOLOv5s | TensorRT | 640×640 | ~30 | ~33ms |
| NX | YOLOv5m | TensorRT | 640×640 | ~20 | ~50ms |

## 编程接口示例

### Python订阅检测结果

```python
#!/usr/bin/env python3
import rospy
from yolov5_ros.msg import DetectionArray

def detection_callback(msg):
    print(f"收到 {msg.detection_count} 个检测结果")
    print(f"推理时间: {msg.inference_time*1000:.1f}ms")
    
    for detection in msg.detections:
        print(f"类别: {detection.class_name}, "
              f"置信度: {detection.confidence:.2f}, "
              f"中心点: ({detection.center.x:.1f}, {detection.center.y:.1f})")

if __name__ == '__main__':
    rospy.init_node('detection_subscriber')
    rospy.Subscriber('/yolov5/detections', DetectionArray, detection_callback)
    rospy.spin()
```

### C++订阅检测结果

```cpp
#include <ros/ros.h>
#include <yolov5_ros/DetectionArray.h>

void detectionCallback(const yolov5_ros::DetectionArray::ConstPtr& msg)
{
    ROS_INFO("收到 %d 个检测结果", msg->detection_count);
    ROS_INFO("推理时间: %.1f ms", msg->inference_time * 1000);
    
    for (const auto& detection : msg->detections)
    {
        ROS_INFO("类别: %s, 置信度: %.2f, 中心点: (%.1f, %.1f)",
                 detection.class_name.c_str(),
                 detection.confidence,
                 detection.center.x,
                 detection.center.y);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "detection_subscriber");
    ros::NodeHandle nh;
    
    ros::Subscriber sub = nh.subscribe("/yolov5/detections", 10, detectionCallback);
    ros::spin();
    
    return 0;
}
```

## 故障排除

### 常见问题

1. **相机无法启动**
   ```bash
   # 检查相机设备
   ls /dev/video*
   # 测试GStreamer管道
   gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
   ```

2. **TensorRT加载失败**
   ```bash
   # 检查TensorRT版本
   python3 -c "import tensorrt as trt; print(trt.__version__)"
   # 检查CUDA版本兼容性
   nvcc --version
   ```

3. **话题无数据**
   ```bash
   # 检查节点状态
   rosnode list
   rosnode info yolov5_detection
   # 检查话题
   rostopic list
   rostopic hz /camera/image_raw
   ```

4. **内存不足**
   - 降低相机分辨率
   - 使用更小的模型 (yolov5n)
   - 减少推理图像尺寸

## 许可证

本项目基于GPL-3.0许可证开源。