#!/usr/bin/env python3
"""
Real-time YOLOv5 detection for NVIDIA TX2/NX with GStreamer camera pipeline and TensorRT acceleration

Usage:
    $ python tx2_realtime_detect.py --weights yolov5s.pt --save-video --save-images
    $ python tx2_realtime_detect.py --weights yolov5s.engine --trt --conf-thres 0.4 --camera-width 1920 --camera-height 1080
    $ python tx2_realtime_detect.py --weights yolov5s.pt --trt --export-trt --trt-workspace 2048
"""

import argparse
import os
import time
import threading
import queue
from pathlib import Path
from datetime import datetime

import cv2
import torch
import numpy as np
import gi

# Initialize GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install TensorRT for acceleration.")

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh, colorstr)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

class TensorRTEngine:
    """TensorRT inference engine for YOLOv5"""
    
    def __init__(self, engine_path, device='cuda:0'):
        """
        Initialize TensorRT engine
        
        Args:
            engine_path (str): Path to TensorRT engine file
            device (str): CUDA device
        """
        self.engine_path = engine_path
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.bindings = []
        self.stream = None
        
        # Load engine
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
            
        # Allocate device memory
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
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            self.bindings.append(int(device_mem))
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data (numpy.ndarray): Input tensor
            
        Returns:
            numpy.ndarray: Output tensor
        """
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize the stream
        self.stream.synchronize()
        
        # Return output
        return self.outputs[0]['host'].copy()

def export_to_tensorrt(weights, imgsz=640, half=False, workspace=4, verbose=False):
    """
    Export YOLOv5 model to TensorRT engine
    
    Args:
        weights (str): Path to PyTorch weights
        imgsz (int): Input image size
        half (bool): Use FP16 precision
        workspace (int): TensorRT workspace size in GB
        verbose (bool): Verbose logging
        
    Returns:
        str: Path to exported engine file
    """
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT not available. Please install TensorRT.")
        
    try:
        import torch
        from export import export_engine
        
        # Export to TensorRT
        engine_path = str(Path(weights).with_suffix('.engine'))
        
        LOGGER.info(f"Starting TensorRT export with workspace={workspace}GB...")
        
        # Load PyTorch model
        device = select_device('0')
        model = DetectMultiBackend(weights, device=device, fp16=half)
        
        # Export
        export_engine(
            model=model.model,
            im=torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())),
            file=Path(weights),
            half=half,
            workspace=workspace,
            verbose=verbose
        )
        
        LOGGER.info(f"TensorRT export complete: {engine_path}")
        return engine_path
        
    except Exception as e:
        LOGGER.error(f"TensorRT export failed: {e}")
        return None

class TX2CameraManager:
    def __init__(self, video_device='/dev/video0', width=1920, height=1080, fps=30):
        """
        Initialize camera manager for TX2/NX with GStreamer
        
        Args:
            video_device (str): Camera device path (not used for nvarguscamerasrc)
            width (int): Camera width in pixels
            height (int): Camera height in pixels  
            fps (int): Camera frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
    def create_camera_pipeline(self):
        """Create GStreamer pipeline for camera capture"""
        pipeline_str = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink sync=false"
        )
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            appsink = self.pipeline.get_by_name("appsink0")
            if appsink is None:
                # Try to get appsink by class
                iterator = self.pipeline.iterate_sinks()
                while True:
                    result, element = iterator.next()
                    if result != Gst.IteratorResult.OK:
                        break
                    if element.get_factory().get_name() == "appsink":
                        appsink = element
                        break
            
            if appsink:
                appsink.set_property("emit-signals", True)
                appsink.connect("new-sample", self._on_new_sample)
            
            return True
        except Exception as e:
            LOGGER.error(f"Failed to create camera pipeline: {e}")
            return False
    
    def _on_new_sample(self, appsink):
        """Callback for new camera frames"""
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get buffer info
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame_data.reshape((self.height, self.width, 3))
                
                # Add frame to queue (drop old frames if queue is full)
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Empty:
                        pass
                
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def start_capture(self):
        """Start camera capture"""
        if not self.create_camera_pipeline():
            return False
        
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True
        LOGGER.info(f"Started camera capture at {self.width}x{self.height}@{self.fps}fps")
        return True
    
    def get_frame(self):
        """Get latest frame from camera"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop camera capture"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            LOGGER.info("Stopped camera capture")

class VideoRecorder:
    def __init__(self, output_dir, width, height, fps=30):
        """Initialize video recorder with GStreamer"""
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.recording = False
        self.pipeline = None
        self.appsrc = None
        
    def start_recording(self, timestamp=None):
        """Start video recording"""
        if self.recording:
            LOGGER.warning("Video recording already in progress")
            return False, None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        video_file = os.path.join(self.output_dir, f"detection_{timestamp}.mp4")
        
        # GStreamer pipeline for video recording
        pipeline_str = (
            f"appsrc name=source ! "
            f"video/x-raw, format=BGR, width={self.width}, height={self.height}, "
            f"framerate={self.fps}/1 ! "
            "videoconvert ! video/x-raw, format=NV12 ! "
            "nvv4l2h264enc insert-sps-pps=true bitrate=8000000 ! "
            "h264parse config-interval=1 ! mp4mux fragment-duration=1000 ! "
            f"filesink location={video_file} sync=false"
        )
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsrc = self.pipeline.get_by_name("source")
            
            if self.appsrc:
                self.appsrc.set_property("format", Gst.Format.TIME)
                self.appsrc.set_property("is-live", True)
            
            self.pipeline.set_state(Gst.State.PLAYING)
            self.recording = True
            LOGGER.info(f"Started video recording to {video_file}")
            return True, video_file
        except Exception as e:
            LOGGER.error(f"Failed to start video recording: {e}")
            return False, None
    
    def write_frame(self, frame):
        """Write frame to video"""
        if not self.recording or not self.appsrc:
            return
        
        try:
            # Convert frame to GStreamer buffer
            frame_bytes = frame.tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
            buffer.fill(0, frame_bytes)
            
            # Push buffer to pipeline
            self.appsrc.emit("push-buffer", buffer)
        except Exception as e:
            LOGGER.error(f"Failed to write frame: {e}")
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.recording:
            return False
        
        try:
            if self.appsrc:
                self.appsrc.emit("end-of-stream")
            
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            self.recording = False
            self.pipeline = None
            self.appsrc = None
            LOGGER.info("Stopped video recording")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to stop video recording: {e}")
            return False

class TX2RealtimeDetector:
    def __init__(self, weights, device='', conf_thres=0.25, iou_thres=0.45, 
                 max_det=1000, imgsz=640, half=False, use_trt=False, 
                 export_trt=False, trt_workspace=4):
        """Initialize YOLOv5 detector for TX2/NX with optional TensorRT acceleration"""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.imgsz = imgsz
        self.use_trt = use_trt
        self.trt_engine = None
        
        # Check if using TensorRT
        if use_trt and TRT_AVAILABLE:
            engine_path = str(Path(weights).with_suffix('.engine'))
            
            # Export to TensorRT if requested or engine doesn't exist
            if export_trt or not os.path.exists(engine_path):
                if weights.endswith('.engine'):
                    LOGGER.error("Cannot export .engine file to TensorRT")
                    self.use_trt = False
                else:
                    LOGGER.info("Exporting model to TensorRT...")
                    exported_engine = export_to_tensorrt(
                        weights=weights,
                        imgsz=imgsz,
                        half=half,
                        workspace=trt_workspace,
                        verbose=True
                    )
                    if exported_engine:
                        engine_path = exported_engine
                    else:
                        LOGGER.warning("TensorRT export failed, falling back to PyTorch")
                        self.use_trt = False
            
            # Load TensorRT engine
            if self.use_trt and os.path.exists(engine_path):
                try:
                    self.trt_engine = TensorRTEngine(engine_path, device)
                    self.names = self._load_class_names(weights)  # Load class names from original model
                    LOGGER.info(f"TensorRT engine loaded successfully: {engine_path}")
                except Exception as e:
                    LOGGER.error(f"Failed to load TensorRT engine: {e}")
                    LOGGER.warning("Falling back to PyTorch inference")
                    self.use_trt = False
        
        # Fallback to PyTorch if TensorRT not available or failed
        if not self.use_trt:
            self.device = select_device(device)
            self.model = DetectMultiBackend(weights, device=self.device, fp16=half)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(imgsz, s=self.stride)
            
            # Warmup
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
            LOGGER.info(f"PyTorch model loaded on {self.device}")
    
    def _load_class_names(self, weights_path):
        """Load class names from original PyTorch model"""
        try:
            if weights_path.endswith('.engine'):
                # For engine files, use COCO names as default
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
                model = DetectMultiBackend(weights_path, device=device)
                return model.names
        except Exception as e:
            LOGGER.warning(f"Failed to load class names: {e}")
            return [f'class{i}' for i in range(80)]  # Default 80 classes
    
    def preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        # Resize frame
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        
        # Convert BGR to RGB and normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        if self.use_trt:
            # For TensorRT, keep as numpy array
            img = img.astype(np.float32) / 255.0
        else:
            # For PyTorch
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.model.fp16 else img.float()
            img /= 255.0
            
            if len(img.shape) == 3:
                img = img[None]  # Add batch dimension
        
        return img
    
    def detect(self, frame):
        """Run detection on frame"""
        # Preprocess
        img = self.preprocess_frame(frame)
        
        if self.use_trt:
            # TensorRT inference
            try:
                # Run inference
                output = self.trt_engine.infer(img)
                
                # Reshape output to match PyTorch format
                # TensorRT output shape: [batch_size, num_detections, 85] for COCO
                output = output.reshape(1, -1, 85)  # Adjust based on your model
                pred = [torch.from_numpy(output[0])]
                
            except Exception as e:
                LOGGER.error(f"TensorRT inference failed: {e}")
                return []
        else:
            # PyTorch inference
            pred = self.model(img)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)
        
        # Process detections
        detections = []
        for det in pred:
            if len(det):
                # Scale boxes back to original frame size
                if self.use_trt:
                    # Convert back to tensor for scale_boxes function
                    det = torch.from_numpy(det) if isinstance(det, np.ndarray) else det
                    
                det[:, :4] = scale_boxes((self.imgsz, self.imgsz), det[:, :4], frame.shape).round()
                detections = det.cpu().numpy() if hasattr(det, 'cpu') else det
        
        return detections
    
    def draw_detections(self, frame, detections, line_thickness=2):
        """Draw detections on frame"""
        annotator = Annotator(frame, line_width=line_thickness, example=str(self.names))
        
        if len(detections):
            for *xyxy, conf, cls in detections:
                c = int(cls)
                if c < len(self.names):
                    label = f'{self.names[c]} {conf:.2f}'
                else:
                    label = f'class{c} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        
        return annotator.result()

def run_realtime_detection(
    weights='yolov5s.pt',
    device='',
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    imgsz=640,
    camera_width=1920,
    camera_height=1080,
    camera_fps=30,
    save_video=False,
    save_images=False,
    output_dir='runs/realtime',
    display=True,
    half=False,
    line_thickness=2,
    use_trt=False,
    export_trt=False,
    trt_workspace=4
):
    """Run real-time detection on TX2/NX with optional TensorRT acceleration"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_dir) / f"detect_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize camera
    camera = TX2CameraManager(width=camera_width, height=camera_height, fps=camera_fps)
    if not camera.start_capture():
        LOGGER.error("Failed to start camera")
        return
    
    # Initialize detector with TensorRT options
    detector = TX2RealtimeDetector(
        weights=weights, device=device, conf_thres=conf_thres,
        iou_thres=iou_thres, max_det=max_det, imgsz=imgsz, half=half,
        use_trt=use_trt, export_trt=export_trt, trt_workspace=trt_workspace
    )
    
    # Print inference backend info
    backend_info = "TensorRT" if detector.use_trt else "PyTorch"
    LOGGER.info(f"Using {colorstr('blue', backend_info)} backend for inference")
    
    # Initialize video recorder
    video_recorder = None
    if save_video:
        video_recorder = VideoRecorder(
            output_dir=str(save_dir), 
            width=camera_width, 
            height=camera_height, 
            fps=camera_fps
        )
        video_recorder.start_recording(timestamp)
    
    # Main detection loop
    frame_count = 0
    start_time = time.time()
    inference_times = []
    
    try:
        LOGGER.info("Starting real-time detection. Press 'q' to quit")
        
        while True:
            # Get frame from camera
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Run detection with timing
            detect_start = time.time()
            detections = detector.detect(frame)
            detect_time = time.time() - detect_start
            inference_times.append(detect_time)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections, line_thickness)
            
            # Save image if requested
            if save_images and frame_count % 30 == 0:  # Save every 30 frames
                img_path = save_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(img_path), result_frame)
            
            # Write to video if recording
            if video_recorder:
                video_recorder.write_frame(result_frame)
            
            # Display frame
            if display:
                # Add performance info to display
                fps_text = f"FPS: {1.0/detect_time:.1f} | Backend: {backend_info}"
                cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                cv2.imshow('TX2 Real-time Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print FPS every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time
                avg_inference = np.mean(inference_times[-100:]) * 1000  # Last 100 frames in ms
                LOGGER.info(f"Processed {frame_count} frames | "
                           f"Overall FPS: {avg_fps:.1f} | "
                           f"Avg inference: {avg_inference:.1f}ms | "
                           f"Backend: {backend_info}")
    
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    
    finally:
        # Cleanup
        camera.stop_capture()
        if video_recorder:
            video_recorder.stop_recording()
        if display:
            cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        total_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_inference = np.mean(inference_times) * 1000 if inference_times else 0
        
        LOGGER.info(f"Detection completed:")
        LOGGER.info(f"  Total frames: {frame_count}")
        LOGGER.info(f"  Average FPS: {total_fps:.1f}")
        LOGGER.info(f"  Average inference time: {avg_inference:.1f}ms")
        LOGGER.info(f"  Backend used: {backend_info}")
        LOGGER.info(f"  Results saved to {colorstr('bold', save_dir)}")

def parse_opt():
    parser = argparse.ArgumentParser(description='TX2/NX Real-time YOLOv5 Detection with TensorRT')
    
    # Model parameters
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model path (.pt or .engine)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    
    # TensorRT parameters
    parser.add_argument('--trt', '--tensorrt', action='store_true', help='use TensorRT for inference')
    parser.add_argument('--export-trt', action='store_true', help='export model to TensorRT engine')
    parser.add_argument('--trt-workspace', type=int, default=4, help='TensorRT workspace size (GB)')
    
    # Camera parameters
    parser.add_argument('--camera-width', type=int, default=1920, help='camera width')
    parser.add_argument('--camera-height', type=int, default=1080, help='camera height')
    parser.add_argument('--camera-fps', type=int, default=30, help='camera fps')
    
    # Output parameters
    parser.add_argument('--save-video', action='store_true', help='save detection video')
    parser.add_argument('--save-images', action='store_true', help='save detection images')
    parser.add_argument('--output-dir', type=str, default='runs/realtime', help='output directory')
    parser.add_argument('--no-display', action='store_true', help='disable display window')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    
    # Check TensorRT availability
    if opt.trt and not TRT_AVAILABLE:
        LOGGER.error("TensorRT requested but not available. Please install TensorRT.")
        LOGGER.info("Install TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html")
        exit(1)
    
    run_realtime_detection(
        weights=opt.weights,
        device=opt.device,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        imgsz=opt.imgsz,
        camera_width=opt.camera_width,
        camera_height=opt.camera_height,
        camera_fps=opt.camera_fps,
        save_video=opt.save_video,
        save_images=opt.save_images,
        output_dir=opt.output_dir,
        display=not opt.no_display,
        half=opt.half,
        line_thickness=opt.line_thickness,
        use_trt=opt.trt,
        export_trt=opt.export_trt,
        trt_workspace=opt.trt_workspace
    )