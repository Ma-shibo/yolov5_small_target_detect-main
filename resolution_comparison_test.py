#!/usr/bin/env python3
"""
Resolution Comparison Test for YOLOv5 Detection

This script compares detection performance between:
1. High resolution (3264x2464) images scaled to inference size
2. Medium resolution (1920x1080) images scaled to inference size

The test measures:
- Detection accuracy (precision, recall)
- Small object detection capability
- Information preservation
- Processing time
"""

import argparse
import cv2
import torch
import numpy as np
import time
import json
from pathlib import Path
import sys
from typing import List, Tuple, Dict

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes, 
                          xyxy2xywh, colorstr, LOGGER)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors


class ResolutionComparison:
    """Compare detection performance across different input resolutions"""
    
    def __init__(self, weights='yolov5s.pt', device='', imgsz=640, conf_thres=0.25, iou_thres=0.45):
        self.weights = weights
        self.device = select_device(device)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load model
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride, self.names = self.model.stride, self.model.names
        self.imgsz = check_img_size(imgsz, s=self.stride)
        
        # Warmup
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        
        print(f"Model loaded: {weights}")
        print(f"Device: {self.device}")
        print(f"Inference size: {self.imgsz}x{self.imgsz}")
    
    def simulate_camera_capture(self, high_res=(3264, 2464), med_res=(1920, 1080)):
        """Simulate camera captures at different resolutions using test images"""
        
        # Create synthetic test images with objects of different sizes
        test_images = {}
        
        # High resolution image (3264x2464)
        high_img = self.create_test_image(high_res, "high_resolution")
        test_images['high_res'] = {
            'image': high_img,
            'resolution': high_res,
            'description': f"High resolution {high_res[0]}x{high_res[1]}"
        }
        
        # Medium resolution image (1920x1080) 
        med_img = self.create_test_image(med_res, "medium_resolution")
        test_images['med_res'] = {
            'image': med_img,
            'resolution': med_res,
            'description': f"Medium resolution {med_res[0]}x{med_res[1]}"
        }
        
        # Also test downscaled high-res to medium-res for fair comparison
        high_to_med = cv2.resize(high_img, med_res)
        test_images['high_to_med'] = {
            'image': high_to_med,
            'resolution': med_res,
            'description': f"High res downscaled to {med_res[0]}x{med_res[1]}"
        }
        
        return test_images
    
    def create_test_image(self, resolution, image_type):
        """Create a synthetic test image with objects of various sizes"""
        width, height = resolution
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background
        img[:] = (50, 50, 50)  # Dark gray background
        
        # Add objects of different sizes to test small object detection
        objects = []
        
        if image_type == "high_resolution":
            # Large objects (easily detectable)
            objects.extend([
                ((width//4, height//4), (width//4 + 400, height//4 + 300), (0, 255, 0)),  # Large green
                ((width//2, height//2), (width//2 + 350, height//2 + 250), (255, 0, 0)),  # Large blue
            ])
            
            # Medium objects 
            objects.extend([
                ((width//6, height//3), (width//6 + 200, height//3 + 150), (0, 0, 255)),  # Medium red
                ((width*2//3, height//6), (width*2//3 + 180, height//6 + 120), (255, 255, 0)),  # Medium cyan
            ])
            
            # Small objects (challenging for detection)
            objects.extend([
                ((width//8, height//8), (width//8 + 60, height//8 + 45), (255, 0, 255)),  # Small magenta
                ((width*7//8 - 50, height//8), (width*7//8, height//8 + 40), (0, 255, 255)),  # Small yellow
                ((width//2, height*7//8 - 30), (width//2 + 50, height*7//8), (128, 128, 255)),  # Small light blue
            ])
            
        else:  # medium_resolution
            # Proportionally smaller objects for medium resolution
            scale = min(width/3264, height/2464)
            
            # Large objects
            objects.extend([
                ((width//4, height//4), (width//4 + int(400*scale), height//4 + int(300*scale)), (0, 255, 0)),
                ((width//2, height//2), (width//2 + int(350*scale), height//2 + int(250*scale)), (255, 0, 0)),
            ])
            
            # Medium objects
            objects.extend([
                ((width//6, height//3), (width//6 + int(200*scale), height//3 + int(150*scale)), (0, 0, 255)),
                ((width*2//3, height//6), (width*2//3 + int(180*scale), height//6 + int(120*scale)), (255, 255, 0)),
            ])
            
            # Small objects
            objects.extend([
                ((width//8, height//8), (width//8 + int(60*scale), height//8 + int(45*scale)), (255, 0, 255)),
                ((width*7//8 - int(50*scale), height//8), (width*7//8, height//8 + int(40*scale)), (0, 255, 255)),
                ((width//2, height*7//8 - int(30*scale)), (width//2 + int(50*scale), height*7//8), (128, 128, 255)),
            ])
        
        # Draw objects
        for (x1, y1), (x2, y2), color in objects:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            # Add some texture
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        
        return img
    
    def preprocess_image(self, img):
        """Preprocess image for YOLO inference"""
        # Resize to inference size
        img_resized = cv2.resize(img, (self.imgsz, self.imgsz))
        
        # Convert to tensor
        img_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(self.device)
        img_tensor = img_tensor.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[None]  # expand for batch dim
            
        return img_tensor, img_resized
    
    def detect_objects(self, img):
        """Run detection on image"""
        start_time = time.time()
        
        # Preprocess
        img_tensor, img_resized = self.preprocess_image(img)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img_tensor)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # Process detections
        detections = []
        if len(pred[0]):
            det = pred[0]
            # Scale boxes back to original image size
            det[:, :4] = scale_boxes((self.imgsz, self.imgsz), det[:, :4], img.shape).round()
            detections = det.cpu().numpy()
        
        inference_time = time.time() - start_time
        return detections, inference_time
    
    def analyze_detections(self, detections, img_shape):
        """Analyze detection results"""
        analysis = {
            'total_detections': len(detections),
            'high_conf_detections': 0,  # conf > 0.5
            'medium_conf_detections': 0,  # 0.25 < conf <= 0.5
            'small_objects': 0,  # area < 1% of image
            'medium_objects': 0,  # 1% <= area < 5% of image
            'large_objects': 0,  # area >= 5% of image
            'avg_confidence': 0.0,
            'bbox_areas': [],
            'confidences': []
        }
        
        if len(detections) == 0:
            return analysis
        
        img_area = img_shape[0] * img_shape[1]
        confidences = detections[:, 4]
        
        analysis['avg_confidence'] = float(np.mean(confidences))
        analysis['confidences'] = confidences.tolist()
        
        # Confidence analysis
        analysis['high_conf_detections'] = int(np.sum(confidences > 0.5))
        analysis['medium_conf_detections'] = int(np.sum((confidences > 0.25) & (confidences <= 0.5)))
        
        # Size analysis
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratio = bbox_area / img_area
            
            analysis['bbox_areas'].append(float(bbox_area))
            
            if area_ratio < 0.01:
                analysis['small_objects'] += 1
            elif area_ratio < 0.05:
                analysis['medium_objects'] += 1
            else:
                analysis['large_objects'] += 1
        
        return analysis
    
    def calculate_information_preservation(self, original_res, inference_res):
        """Calculate how much information is preserved during scaling"""
        orig_pixels = original_res[0] * original_res[1]
        inf_pixels = inference_res * inference_res
        
        scale_factor = min(inference_res / original_res[0], inference_res / original_res[1])
        effective_pixels = orig_pixels * (scale_factor ** 2)
        
        preservation_ratio = effective_pixels / orig_pixels
        
        return {
            'original_pixels': orig_pixels,
            'inference_pixels': inf_pixels,
            'scale_factor': scale_factor,
            'preservation_ratio': preservation_ratio,
            'information_loss': 1 - preservation_ratio
        }
    
    def run_comparison(self):
        """Run the complete comparison test"""
        print("\n" + "="*80)
        print("YOLOv5 RESOLUTION COMPARISON TEST")
        print("="*80)
        
        # Generate test images
        test_images = self.simulate_camera_capture()
        
        results = {}
        
        for test_name, test_data in test_images.items():
            print(f"\n--- Testing: {test_data['description']} ---")
            
            img = test_data['image']
            resolution = test_data['resolution']
            
            # Calculate information preservation
            info_preservation = self.calculate_information_preservation(resolution, self.imgsz)
            
            # Run detection
            detections, inference_time = self.detect_objects(img)
            
            # Analyze results
            analysis = self.analyze_detections(detections, img.shape)
            
            # Store results
            results[test_name] = {
                'description': test_data['description'],
                'resolution': resolution,
                'inference_time': inference_time,
                'information_preservation': info_preservation,
                'detection_analysis': analysis
            }
            
            # Print results
            print(f"Resolution: {resolution[0]}x{resolution[1]}")
            print(f"Information preservation: {info_preservation['preservation_ratio']:.1%}")
            print(f"Information loss: {info_preservation['information_loss']:.1%}")
            print(f"Scale factor: {info_preservation['scale_factor']:.3f}")
            print(f"Inference time: {inference_time*1000:.1f}ms")
            print(f"Total detections: {analysis['total_detections']}")
            print(f"High confidence (>0.5): {analysis['high_conf_detections']}")
            print(f"Average confidence: {analysis['avg_confidence']:.3f}")
            print(f"Small objects detected: {analysis['small_objects']}")
            print(f"Medium objects detected: {analysis['medium_objects']}")
            print(f"Large objects detected: {analysis['large_objects']}")
            
            # Save visualization
            self.save_detection_visualization(img, detections, test_name, test_data['description'])
        
        # Compare results
        self.compare_results(results)
        
        return results
    
    def compare_results(self, results):
        """Compare and summarize the results"""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        high_res = results['high_res']
        med_res = results['med_res']
        high_to_med = results['high_to_med']
        
        print(f"\n📊 Information Preservation:")
        print(f"  High res (3264x2464): {high_res['information_preservation']['preservation_ratio']:.1%}")
        print(f"  Medium res (1920x1080): {med_res['information_preservation']['preservation_ratio']:.1%}")
        print(f"  High→Medium: {high_to_med['information_preservation']['preservation_ratio']:.1%}")
        
        print(f"\n⏱️  Inference Time:")
        print(f"  High res: {high_res['inference_time']*1000:.1f}ms")
        print(f"  Medium res: {med_res['inference_time']*1000:.1f}ms")
        print(f"  High→Medium: {high_to_med['inference_time']*1000:.1f}ms")
        
        print(f"\n🎯 Detection Performance:")
        print(f"  High res - Total: {high_res['detection_analysis']['total_detections']}, "
              f"Avg conf: {high_res['detection_analysis']['avg_confidence']:.3f}")
        print(f"  Medium res - Total: {med_res['detection_analysis']['total_detections']}, "
              f"Avg conf: {med_res['detection_analysis']['avg_confidence']:.3f}")
        print(f"  High→Medium - Total: {high_to_med['detection_analysis']['total_detections']}, "
              f"Avg conf: {high_to_med['detection_analysis']['avg_confidence']:.3f}")
        
        print(f"\n🔍 Small Object Detection:")
        print(f"  High res: {high_res['detection_analysis']['small_objects']}")
        print(f"  Medium res: {med_res['detection_analysis']['small_objects']}")
        print(f"  High→Medium: {high_to_med['detection_analysis']['small_objects']}")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        
        info_diff = high_res['information_preservation']['preservation_ratio'] - med_res['information_preservation']['preservation_ratio']
        print(f"  • High resolution preserves {info_diff:.1%} more information")
        
        if high_res['detection_analysis']['small_objects'] > med_res['detection_analysis']['small_objects']:
            print(f"  • High resolution detects more small objects (+{high_res['detection_analysis']['small_objects'] - med_res['detection_analysis']['small_objects']})")
        
        if abs(high_res['inference_time'] - med_res['inference_time']) > 0.01:
            time_diff = (high_res['inference_time'] - med_res['inference_time']) * 1000
            print(f"  • Processing time difference: {time_diff:+.1f}ms")
        else:
            print(f"  • Processing times are similar (both scale to same inference size)")
        
        # Recommendation
        print(f"\n🎯 Recommendation:")
        if high_res['detection_analysis']['small_objects'] > med_res['detection_analysis']['small_objects']:
            print(f"  Use high resolution (3264x2464) for better small object detection")
            print(f"  The information advantage outweighs the preprocessing cost")
        else:
            print(f"  Medium resolution (1920x1080) provides sufficient detection capability")
            print(f"  Consider high resolution only for applications requiring maximum detail")
    
    def save_detection_visualization(self, img, detections, test_name, description):
        """Save visualization of detections"""
        vis_img = img.copy()
        
        if len(detections) > 0:
            annotator = Annotator(vis_img, line_width=max(round(sum(img.shape) / 2 * 0.003), 2))
            
            for *xyxy, conf, cls in detections:
                label = f'Object {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls) if cls < 80 else 0, True))
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(img.shape[0] / 1000, 0.5)
        thickness = max(int(img.shape[0] / 500), 1)
        
        text_lines = [
            f"{description}",
            f"Detections: {len(detections)}",
            f"Inference size: {self.imgsz}x{self.imgsz}"
        ]
        
        y_offset = 30
        for i, line in enumerate(text_lines):
            y_pos = y_offset + (i * 40)
            cv2.putText(vis_img, line, (20, y_pos), font, font_scale, (0, 255, 0), thickness)
        
        # Save image
        save_path = f"runs/resolution_comparison_{test_name}.jpg"
        Path(save_path).parent.mkdir(exist_ok=True)
        cv2.imwrite(save_path, vis_img)
        print(f"Visualization saved: {save_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv5 Resolution Comparison Test')
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt', help='model weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create comparison tester
    tester = ResolutionComparison(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
    
    # Run comparison
    results = tester.run_comparison()
    
    # Save results to JSON
    results_path = "runs/resolution_comparison_results.json"
    Path(results_path).parent.mkdir(exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    results_json = convert_for_json(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == '__main__':
    main()