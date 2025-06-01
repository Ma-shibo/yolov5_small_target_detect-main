#!/usr/bin/env python3
"""
YOLOv5 ROS Detection Node with TensorRT Support

This node subscribes to image topics and publishes detection results.
Supports both PyTorch and TensorRT backends for inference acceleration.
"""

import rospy
import cv2
import torch
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add YOLOv5 to path
YOLO_PATH = Path(__file__).parent.parent.parent.absolute()
if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

# ROS imports
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

# Custom ROS messages
from yolov5_ros.msg import Detection, DetectionArray, BoundingBox

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, colorstr)
from utils.torch_utils import select_device

# TensorRT imports (optional)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    LOGGER.warning("TensorRT not available. Using PyTorch backend only.")


class TensorRTEngine:
    """TensorRT inference engine for YOLOv5"""
    
    def __init__(self, engine_path, device='cuda:0'):
        self.engine_path = engine_path
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.bindings = []
        self.stream = None
        
        self._load_engine()
        
    def _load_engine(self):
        """Load TensorRT engine from file"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
            
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
            
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine")
            
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context")
            
        self.stream = cuda.Stream()
        self._allocate_buffers()
        
        LOGGER.info(f"TensorRT engine loaded: {self.engine_path}")
        
    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, input_data):
        """Run inference on input data"""
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        self.stream.synchronize()
        
        return self.outputs[0]['host'].copy()


class YOLOv5Detector:
    """YOLOv5 detector with ROS integration"""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolov5_detection_node', anonymous=True)
        
        # Get parameters from ROS parameter server
        self.weights = rospy.get_param('~weights', 'yolov5s.pt')
        self.device = rospy.get_param('~device', '')
        self.conf_thres = rospy.get_param('~conf_thres', 0.25)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        self.max_det = rospy.get_param('~max_det', 1000)
        self.imgsz = rospy.get_param('~imgsz', 640)
        self.half = rospy.get_param('~half', False)
        self.use_trt = rospy.get_param('~use_trt', False)
        self.agnostic_nms = rospy.get_param('~agnostic_nms', False)
        self.classes = rospy.get_param('~classes', None)
        
        # Image processing parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.detection_topic = rospy.get_param('~detection_topic', '/yolov5/detections')
        self.visualize = rospy.get_param('~visualize', False)  # 默认关闭可视化
        self.pub_image = rospy.get_param('~publish_image', False)  # 默认不发布图像
        self.save_debug_images = rospy.get_param('~save_debug_images', False)  # 可选保存调试图像
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize detector
        self._init_detector()
        
        # Setup ROS publishers and subscribers
        self.detection_pub = rospy.Publisher(self.detection_topic, DetectionArray, queue_size=10)
        
        if self.pub_image:
            self.image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=10)
            
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        
        # Statistics
        self.detection_count = 0
        self.total_inference_time = 0.0
        
        rospy.loginfo(f"YOLOv5 ROS node initialized")
        rospy.loginfo(f"Weights: {self.weights}")
        rospy.loginfo(f"Backend: {'TensorRT' if self.use_trt else 'PyTorch'}")
        rospy.loginfo(f"Image topic: {self.image_topic}")
        rospy.loginfo(f"Detection topic: {self.detection_topic}")
        
    def _init_detector(self):
        """Initialize YOLOv5 detector"""
        self.use_trt = self.use_trt and TRT_AVAILABLE
        self.trt_engine = None
        
        if self.use_trt:
            engine_path = str(Path(self.weights).with_suffix('.engine'))
            
            if os.path.exists(engine_path):
                try:
                    self.trt_engine = TensorRTEngine(engine_path, self.device)
                    self.names = self._load_class_names()
                    rospy.loginfo(f"TensorRT engine loaded: {engine_path}")
                except Exception as e:
                    rospy.logerr(f"Failed to load TensorRT engine: {e}")
                    rospy.logwarn("Falling back to PyTorch backend")
                    self.use_trt = False
            else:
                rospy.logwarn(f"TensorRT engine not found: {engine_path}")
                rospy.logwarn("Falling back to PyTorch backend")
                self.use_trt = False
        
        if not self.use_trt:
            # Load PyTorch model
            self.device = select_device(self.device)
            self.model = DetectMultiBackend(self.weights, device=self.device, fp16=self.half)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            
            # Warmup
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
            rospy.loginfo(f"PyTorch model loaded on {self.device}")
    
    def _load_class_names(self):
        """Load class names for TensorRT engine"""
        try:
            if self.weights.endswith('.engine'):
                # Use COCO names as default for engine files
                return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                       'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                       'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                       'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                       'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                       'toothbrush']
            else:
                # Load from PyTorch model
                device = select_device('')
                model = DetectMultiBackend(self.weights, device=device)
                return model.names
        except Exception as e:
            rospy.logwarn(f"Failed to load class names: {e}")
            return [f'class{i}' for i in range(80)]
    
    def preprocess_image(self, cv_image):
        """Preprocess image for detection"""
        # Resize image
        img = cv2.resize(cv_image, (self.imgsz, self.imgsz))
        
        # Convert BGR to RGB and normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        if self.use_trt:
            # For TensorRT
            img = img.astype(np.float32) / 255.0
        else:
            # For PyTorch
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.model.fp16 else img.float()
            img /= 255.0
            
            if len(img.shape) == 3:
                img = img[None]  # Add batch dimension
        
        return img
    
    def detect(self, cv_image):
        """Run detection on image"""
        start_time = time.time()
        
        # Preprocess
        img = self.preprocess_image(cv_image)
        
        if self.use_trt:
            # TensorRT inference
            try:
                output = self.trt_engine.infer(img)
                output = output.reshape(1, -1, 85)  # Adjust based on model output
                pred = [torch.from_numpy(output[0])]
            except Exception as e:
                rospy.logerr(f"TensorRT inference failed: {e}")
                return [], time.time() - start_time
        else:
            # PyTorch inference
            pred = self.model(img)
        
        # NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, 
            classes=self.classes, agnostic=self.agnostic_nms, max_det=self.max_det
        )
        
        # Process detections
        detections = []
        for det in pred:
            if len(det):
                if self.use_trt:
                    det = torch.from_numpy(det) if isinstance(det, np.ndarray) else det
                    
                det[:, :4] = scale_boxes((self.imgsz, self.imgsz), det[:, :4], cv_image.shape).round()
                detections = det.cpu().numpy() if hasattr(det, 'cpu') else det
        
        inference_time = time.time() - start_time
        return detections, inference_time
    
    def create_detection_msg(self, detections, cv_image, inference_time, header):
        """Create ROS detection message (optimized - no image transmission)"""
        detection_array = DetectionArray()
        detection_array.header = header
        detection_array.inference_time = inference_time
        detection_array.detection_count = len(detections)
        
        # Add image dimensions instead of full image
        detection_array.image_width = cv_image.shape[1]
        detection_array.image_height = cv_image.shape[0]
        
        # Process each detection
        for *xyxy, conf, cls in detections:
            detection_msg = Detection()
            detection_msg.header = header
            
            # Class information
            class_id = int(cls)
            detection_msg.class_id = class_id
            detection_msg.class_name = self.names[class_id] if class_id < len(self.names) else f'class{class_id}'
            detection_msg.confidence = float(conf)
            
            # Bounding box
            x1, y1, x2, y2 = map(float, xyxy)
            detection_msg.bbox.x = x1
            detection_msg.bbox.y = y1
            detection_msg.bbox.width = x2 - x1
            detection_msg.bbox.height = y2 - y1
            
            # Center point
            detection_msg.center.x = (x1 + x2) / 2
            detection_msg.center.y = (y1 + y2) / 2
            detection_msg.center.z = 0.0
            
            detection_array.detections.append(detection_msg)
        
        # 可选：保存调试图像到本地而不是通过话题传输
        if self.save_debug_images and len(detections) > 0:
            self._save_debug_image(cv_image, detections, header.stamp)
        
        return detection_array
    
    def _save_debug_image(self, cv_image, detections, timestamp):
        """Save debug image with detections to local file"""
        try:
            debug_dir = Path("/tmp/yolov5_debug")
            debug_dir.mkdir(exist_ok=True)
            
            result_image = self.visualize_detections(cv_image, detections)
            filename = f"detection_{timestamp.secs}_{timestamp.nsecs}.jpg"
            filepath = debug_dir / filename
            
            # 保存缩小版本以节省空间
            height, width = result_image.shape[:2]
            if width > 1920:  # 如果宽度超过1920，等比例缩放
                scale = 1920.0 / width
                new_width = 1920
                new_height = int(height * scale)
                result_image = cv2.resize(result_image, (new_width, new_height))
            
            cv2.imwrite(str(filepath), result_image)
            rospy.logdebug(f"Debug image saved: {filepath}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to save debug image: {e}")
    
    def visualize_detections(self, cv_image, detections):
        """Draw detections on image"""
        if len(detections) == 0:
            return cv_image
        
        result_image = cv_image.copy()
        
        for *xyxy, conf, cls in detections:
            # Get class info
            class_id = int(cls)
            class_name = self.names[class_id] if class_id < len(self.names) else f'class{class_id}'
            label = f'{class_name} {conf:.2f}'
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 255, 0)  # Green color
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def image_callback(self, msg):
        """Process incoming image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return
        
        # Run detection
        detections, inference_time = self.detect(cv_image)
        
        # Update statistics
        self.detection_count += 1
        self.total_inference_time += inference_time
        
        # Create and publish detection message
        detection_msg = self.create_detection_msg(detections, cv_image, inference_time, msg.header)
        self.detection_pub.publish(detection_msg)
        
        # Publish visualization if enabled
        if self.pub_image and self.visualize:
            result_image = self.visualize_detections(cv_image, detections)
            try:
                result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                result_msg.header = msg.header
                self.image_pub.publish(result_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CV Bridge error: {e}")
        
        # Log performance
        if self.detection_count % 50 == 0:
            avg_time = self.total_inference_time / self.detection_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
            rospy.loginfo(f"Processed {self.detection_count} frames | "
                         f"Avg inference: {avg_time*1000:.1f}ms | "
                         f"FPS: {fps:.1f} | "
                         f"Detections: {len(detections)}")
    
    def run(self):
        """Main loop"""
        rospy.loginfo("YOLOv5 detection node started")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down YOLOv5 detection node")
        finally:
            # Print final statistics
            if self.detection_count > 0:
                avg_time = self.total_inference_time / self.detection_count
                rospy.loginfo(f"Final statistics:")
                rospy.loginfo(f"  Total frames processed: {self.detection_count}")
                rospy.loginfo(f"  Average inference time: {avg_time*1000:.1f}ms")
                rospy.loginfo(f"  Average FPS: {1.0/avg_time:.1f}")


if __name__ == '__main__':
    try:
        detector = YOLOv5Detector()
        detector.run()
    except rospy.ROSInterruptException:
        pass