#!/usr/bin/env python3
"""
YOLOv5 ROS Camera Node for TX2/NX

This node publishes camera images using GStreamer hardware acceleration.
Optimized for NVIDIA TX2/NX platforms.
"""

import rospy
import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# ROS imports
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError


class TX2CameraNode:
    """TX2/NX Camera node with GStreamer hardware acceleration"""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolov5_camera_node', anonymous=True)
        
        # Get parameters from ROS parameter server
        self.camera_width = rospy.get_param('~width', 1920)
        self.camera_height = rospy.get_param('~height', 1080)
        self.camera_fps = rospy.get_param('~fps', 30)
        self.sensor_id = rospy.get_param('~sensor_id', 0)
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.frame_id = rospy.get_param('~frame_id', 'camera_link')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Setup publishers
        self.image_pub = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher(self.camera_info_topic, CameraInfo, queue_size=10)
        
        # Camera pipeline
        self.pipeline = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        rospy.loginfo(f"TX2 Camera node initialized")
        rospy.loginfo(f"Resolution: {self.camera_width}x{self.camera_height}@{self.camera_fps}fps")
        rospy.loginfo(f"Image topic: {self.image_topic}")
        rospy.loginfo(f"Camera info topic: {self.camera_info_topic}")
        
    def create_camera_pipeline(self):
        """Create GStreamer pipeline for camera capture"""
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.camera_width}, height={self.camera_height}, "
            f"format=NV12, framerate={self.camera_fps}/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink sync=false"
        )
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Get appsink element
            appsink = None
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
            rospy.logerr(f"Failed to create camera pipeline: {e}")
            return False
    
    def _on_new_sample(self, appsink):
        """Callback for new camera frames"""
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            
            # Get buffer info
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame_data.reshape((self.camera_height, self.camera_width, 3))
                
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
    
    def start_camera(self):
        """Start camera capture"""
        if not self.create_camera_pipeline():
            return False
        
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True
        rospy.loginfo(f"Started camera capture")
        return True
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            rospy.loginfo("Stopped camera capture")
    
    def get_frame(self):
        """Get latest frame from camera"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def create_camera_info_msg(self, header):
        """Create camera info message with basic parameters"""
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width = self.camera_width
        camera_info.height = self.camera_height
        
        # Basic camera matrix (you should calibrate your camera for accurate values)
        camera_info.K = [
            self.camera_width * 0.8, 0, self.camera_width / 2,
            0, self.camera_height * 0.8, self.camera_height / 2,
            0, 0, 1
        ]
        
        # Distortion coefficients (assuming no distortion)
        camera_info.D = [0, 0, 0, 0, 0]
        
        # Rectification matrix (identity for monocular camera)
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        
        # Projection matrix
        camera_info.P = [
            camera_info.K[0], 0, camera_info.K[2], 0,
            0, camera_info.K[4], camera_info.K[5], 0,
            0, 0, 1, 0
        ]
        
        camera_info.distortion_model = "plumb_bob"
        
        return camera_info
    
    def publish_frame(self):
        """Main publishing loop"""
        rate = rospy.Rate(self.camera_fps)
        
        while not rospy.is_shutdown() and self.is_running:
            frame = self.get_frame()
            if frame is not None:
                try:
                    # Create header with timestamp
                    header = Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = self.frame_id
                    
                    # Convert OpenCV image to ROS message
                    image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    image_msg.header = header
                    
                    # Create camera info message
                    camera_info_msg = self.create_camera_info_msg(header)
                    
                    # Publish messages
                    self.image_pub.publish(image_msg)
                    self.camera_info_pub.publish(camera_info_msg)
                    
                    # Update statistics
                    self.frame_count += 1
                    
                    # Log performance every 100 frames
                    if self.frame_count % 100 == 0:
                        elapsed_time = time.time() - self.start_time
                        fps = self.frame_count / elapsed_time
                        rospy.loginfo(f"Published {self.frame_count} frames | FPS: {fps:.1f}")
                
                except CvBridgeError as e:
                    rospy.logerr(f"CV Bridge error: {e}")
            
            rate.sleep()
    
    def run(self):
        """Main execution function"""
        if not self.start_camera():
            rospy.logerr("Failed to start camera")
            return
        
        try:
            # Start publishing in a separate thread
            publish_thread = threading.Thread(target=self.publish_frame)
            publish_thread.daemon = True
            publish_thread.start()
            
            rospy.loginfo("Camera node started, publishing frames...")
            rospy.spin()
            
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down camera node")
        finally:
            self.stop_camera()
            
            # Print final statistics
            if self.frame_count > 0:
                elapsed_time = time.time() - self.start_time
                avg_fps = self.frame_count / elapsed_time
                rospy.loginfo(f"Final statistics:")
                rospy.loginfo(f"  Total frames published: {self.frame_count}")
                rospy.loginfo(f"  Average FPS: {avg_fps:.1f}")


if __name__ == '__main__':
    try:
        camera_node = TX2CameraNode()
        camera_node.run()
    except rospy.ROSInterruptException:
        pass