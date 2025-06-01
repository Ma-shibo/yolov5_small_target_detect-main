#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 小目标检测视频测试脚本
使用训练好的权重文件对视频进行检测 - 单视频详细测试版本
支持时间范围检测和ROI区域设置
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from datetime import datetime, timedelta
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 向上两级到yolov5-v7目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def time_to_seconds(time_str):
    """将时间字符串转换为秒数"""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(parts[0])
    except:
        return 0

def apply_roi(image, roi_params, roi_mode='percentage'):
    """应用ROI区域裁剪"""
    h, w = image.shape[:2]
    
    if roi_mode == 'percentage':
        x_percent, y_percent, width_percent, height_percent = roi_params
        x = int(w * x_percent / 100)
        y = int(h * y_percent / 100)
        width = int(w * width_percent / 100)
        height = int(h * height_percent / 100)
    else:  # absolute
        x, y, width, height = roi_params
        
    # 确保ROI在图像范围内
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    # 裁剪图像
    roi_image = image[y:y+height, x:x+width]
    
    return roi_image, (x, y, width, height)

def get_first_video(source_dir):
    """获取目录中的第一个视频文件或图片文件"""
    source_path = Path(source_dir)
    if source_path.is_file():
        return source_path
    
    # 首先搜索视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    for ext in video_extensions:
        videos = list(source_path.glob(f'*{ext}'))
        if videos:
            return videos[0]  # 返回第一个找到的视频
    
    # 如果没有找到视频文件，搜索图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP']
    for ext in image_extensions:
        images = list(source_path.glob(f'*{ext}'))
        if images:
            # 对于图片，返回整个目录路径，让LoadImages处理所有图片
            return source_path
    
    raise FileNotFoundError(f"在目录 {source_dir} 中没有找到视频文件或图片文件")

@smart_inference_mode()
def run_single_video_test(
        weights='runs/small_target_train/exp7/weights/best.pt',  # 强制使用指定权重
        source='/home/lkx/Videos/2025_CUADC_FORWARD/',
        data=None,
        imgsz=(640, 640),
        conf_thres=0.15,  # 降低置信度阈值以检测更多目标
        iou_thres=0.45,
        max_det=1000,
        device='0',
        view_img=False,
        save_txt=True,      # 保存标签文件
        save_conf=True,     # 保存置信度
        save_crop=True,     # 保存裁剪区域
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,    # 关闭可视化以提高速度
        update=False,
        project='runs/detect',
        name='single_video_detailed_test',
        exist_ok=True,
        line_thickness=2,   # 细一点的线条
        hide_labels=False,
        hide_conf=False,
        half=True,          # 使用半精度推理提高速度
        dnn=False,
        vid_stride=1,
        max_frames=3000,     # 限制最大测试帧数
        start_time=None,     # 开始时间
        end_time=None,       # 结束时间
        save_as_images=False, # 保存为图片
        roi_params=None,     # ROI参数
        roi_mode='percentage', # ROI模式
):
    """运行单视频详细检测测试"""
    
    # 确保使用指定的权重文件
    if not Path(weights).exists():
        raise FileNotFoundError(f"指定的权重文件不存在: {weights}")
    
    print(f"🎯 强制使用权重文件: {weights}")
    
    # 获取第一个视频文件或图片文件夹
    first_video = get_first_video(source)
    source = str(first_video)
    
    # 检查是否为图片文件夹
    is_image_folder = Path(source).is_dir() and any(
        Path(source).glob(f'*{ext}') 
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP']
    )
    
    if is_image_folder:
        print(f"📁 测试图片文件夹: {Path(source).name}")
        print(f"⚡ 优化设置: 半精度推理={half}, 最大处理图片数={max_frames}")
    else:
        print(f"📹 测试视频: {Path(source).name}")
        print(f"⚡ 优化设置: 半精度推理={half}, 最大测试帧数={max_frames}")
    
    # 时间范围设置（仅对视频有效）
    start_seconds = time_to_seconds(start_time) if start_time and not is_image_folder else 0
    end_seconds = time_to_seconds(end_time) if end_time and not is_image_folder else None
    
    if (start_time or end_time) and not is_image_folder:
        print(f"⏰ 时间范围: {start_time or '开始'} 到 {end_time or '结束'}")
    
    # ROI设置
    if roi_params:
        print(f"🎯 ROI设置: {roi_mode} 模式 - {roi_params}")
    
    # 创建详细的保存目录结构
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
    (save_dir / 'crops').mkdir(parents=True, exist_ok=True)
    (save_dir / 'statistics').mkdir(parents=True, exist_ok=True)
    
    if save_as_images:
        (save_dir / 'images').mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    
    # 修复imgsz处理 - 确保是正确的格式
    if isinstance(imgsz, (int, float)):
        imgsz = (int(imgsz), int(imgsz))
    elif isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            imgsz = (int(imgsz[0]), int(imgsz[0]))
        elif len(imgsz) == 2:
            imgsz = (int(imgsz[0]), int(imgsz[1]))
        else:
            imgsz = (int(imgsz[0]), int(imgsz[0]))
    
    imgsz = check_img_size(imgsz, s=stride)

    print(f"📊 模型类别: {names}")
    print(f"🔧 图像尺寸: {imgsz}")
    print(f"💻 设备: {device}")
    print(f"⚡ 半精度: {half}")

    # 数据加载
    bs = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理 - 修复预热参数
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        warmup_size = (1 if pt or model.triton else bs, 3, imgsz[0], imgsz[1])
    else:
        warmup_size = (1 if pt or model.triton else bs, 3, imgsz, imgsz)
    
    model.warmup(imgsz=warmup_size)
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    # 详细统计数据
    detailed_results = []
    frame_statistics = {}
    detection_summary = {}
    
    # 初始化类别统计
    for class_id, class_name in names.items():
        detection_summary[class_name] = {'count': 0, 'total_conf': 0.0, 'avg_conf': 0.0}
    
    if is_image_folder:
        print(f"\n🚀 开始处理图片文件夹...")
    else:
        print(f"\n🚀 开始处理视频...")
    
    start_time_process = time.time()
    
    frame_count = 0
    processed_frames = 0
    
    # 使用LoadImages迭代处理图片或视频帧
    for path, im, im0s, vid_cap, s in dataset:
        # 检查帧数限制
        if processed_frames >= max_frames:
            print(f"⏹️  达到最大处理限制: {max_frames}")
            break
        
        # 对于图片文件夹，不需要视频相关的时间检查
        if not is_image_folder and vid_cap is not None:
            current_frame_num = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            
            # 获取视频信息（仅对视频）
            if processed_frames == 0:
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps
                start_frame = int(start_seconds * fps) if start_seconds else 0
                end_frame = int(end_seconds * fps) if end_seconds else total_frames
                
                # 跳转到起始帧
                if start_frame > 0:
                    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    print(f"⏭️  跳转到起始帧: {start_frame}")
                
                print(f"📊 视频信息: {total_frames} 帧, {fps:.2f} FPS, 时长 {duration:.2f} 秒")
            
            # 检查是否超出时间范围
            if end_seconds and current_frame_num >= end_seconds * fps:
                print(f"⏹️  达到结束时间")
                break
        else:
            # 对于图片，使用处理计数作为"帧号"
            current_frame_num = processed_frames
            fps = 1  # 图片没有fps概念
            
            # 对于图片文件夹，设置合理的默认值
            if processed_frames == 0:
                start_frame = 0
                end_frame = max_frames
                print(f"📊 图片处理: 最大处理数量 {max_frames}")
        
        # 跳帧处理（主要针对视频）
        if not is_image_folder and frame_count % vid_stride != 0:
            frame_count += 1
            continue
        
        # 应用ROI
        original_frame = im0s.copy()
        roi_offset = (0, 0, 0, 0)  # x, y, width, height
        
        if roi_params:
            im0s, roi_offset = apply_roi(im0s, roi_params, roi_mode)
            if im0s.size == 0:
                print(f"⚠️  ROI区域无效，跳过 {path}")
                frame_count += 1
                continue
        
        # 预处理图像
        im0 = im0s.copy()
        
        # 获取原始图像尺寸
        h0, w0 = im0s.shape[:2]  # 原始高度和宽度
        
        # 修复图像尺寸处理 - 确保正确的缩放比例
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) == 2:
                target_h, target_w = int(imgsz[0]), int(imgsz[1])
            else:
                target_h = target_w = int(imgsz[0])
        else:
            target_h = target_w = int(imgsz)
        
        # 保持宽高比的缩放
        r = min(target_h / h0, target_w / w0)  # 缩放比例
        new_h, new_w = int(h0 * r), int(w0 * r)
        
        # 计算填充
        dh, dw = target_h - new_h, target_w - new_w  # 高度和宽度差
        dh /= 2  # 平分填充
        dw /= 2
        
        # 缩放图像
        if (h0, w0) != (new_h, new_w):
            im = cv2.resize(im0s, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            im = im0s
        
        # 填充图像到目标尺寸
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 存储缩放信息用于后续坐标转换
        ratio_pad = ((r, r), (dw, dh))  # 缩放比例和填充
        
        # 转换为模型输入格式
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # 推理
        with dt[1]:
            pred = model(im, augment=augment, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 处理预测结果
        for i, det in enumerate(pred):
            seen += 1
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            # 当前帧的检测信息
            frame_detections = []
            
            if len(det):
                # 使用正确的坐标转换 - 从模型输入尺寸转换到原始图像尺寸
                # 这里需要使用我们之前计算的ratio_pad信息
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape, ratio_pad).round()
                
                # 如果使用了ROI，需要调整坐标到原始图像
                if roi_params:
                    det[:, :4][:, [0, 2]] += roi_offset[0]  # x坐标
                    det[:, :4][:, [1, 3]] += roi_offset[1]  # y坐标

                # 处理每个检测
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    class_name = names[c]
                    confidence = float(conf)
                    
                    # 更新统计
                    detection_summary[class_name]['count'] += 1
                    detection_summary[class_name]['total_conf'] += confidence
                    
                    # 保存详细检测信息
                    detection_info = {
                        'class_id': c,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [float(x) for x in xyxy],
                        'frame': current_frame_num,
                        'time_seconds': current_frame_num / fps
                    }
                    frame_detections.append(detection_info)
                    
                    # 保存标签文件（包含置信度）
                    if save_txt:
                        # 根据输入类型确定标签文件名
                        if is_image_folder:
                            # 对于图片文件夹，使用原始图片文件名
                            label_name = Path(path).stem + '.txt'
                        else:
                            # 对于视频，使用视频名+帧数格式
                            video_name = Path(source).stem
                            label_name = f"{video_name}_frame_{current_frame_num:06d}.txt"
                        
                        txt_path = save_dir / 'labels' / label_name
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 保存裁剪区域
                    if save_crop and detection_summary[class_name]['count'] <= 100:
                        crop_dir = save_dir / 'crops' / class_name
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 根据输入类型确定裁剪文件名
                        if is_image_folder:
                            # 对于图片文件夹，使用原始图片文件名
                            crop_name = f"{Path(path).stem}_{confidence:.3f}.jpg"
                        else:
                            # 对于视频，使用视频名+帧数格式
                            video_name = Path(source).stem
                            crop_name = f"{video_name}_frame_{current_frame_num:06d}_{confidence:.3f}.jpg"
                        
                        # 使用原始图像进行裁剪
                        original_xyxy = xyxy if not roi_params else [
                            xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        ]
                        save_one_box(original_xyxy, original_frame, 
                                   file=crop_dir / crop_name, BGR=True)

                    # 绘制检测框
                    if not (hide_labels or hide_conf):
                        label = f'{class_name} {confidence:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # 保存帧统计
            frame_statistics[current_frame_num] = {
                'detections_count': len(det) if len(det) else 0,
                'inference_time': dt[1].dt * 1000,  # ms
                'detections': frame_detections,
                'time_seconds': current_frame_num / fps
            }

            # 保存结果图像
            im0 = annotator.result()
            
            # 在图像上添加帧信息
            info_text = f"Frame: {current_frame_num} | Time: {current_frame_num/fps:.1f}s | Detections: {len(det) if len(det) else 0}"
            cv2.putText(im0, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 如果启用ROI，在原始图像上显示ROI区域
            if roi_params:
                display_frame = original_frame.copy()
                # 在原始图像上绘制ROI区域
                cv2.rectangle(display_frame, (roi_offset[0], roi_offset[1]), 
                            (roi_offset[0] + roi_offset[2], roi_offset[1] + roi_offset[3]), 
                            (0, 255, 255), 2)
                cv2.putText(display_frame, "ROI", (roi_offset[0], roi_offset[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 将检测结果叠加到ROI区域
                roi_resized = cv2.resize(im0, (roi_offset[2], roi_offset[3]))
                display_frame[roi_offset[1]:roi_offset[1]+roi_offset[3], 
                            roi_offset[0]:roi_offset[0]+roi_offset[2]] = roi_resized
                im0 = display_frame

            # 保存图像或视频
            if is_image_folder or save_as_images:
                # 对于图片文件夹，保存为单独的图片，使用原始文件名
                if is_image_folder:
                    img_name = Path(path).stem + '_detected.jpg'
                else:
                    # 对于视频保存为图片，使用视频名+帧数格式
                    video_name = Path(source).stem
                    img_name = f"{video_name}_frame_{current_frame_num:06d}_detected.jpg"
                
                img_path = save_dir / 'images' / img_name
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_path), im0)
            elif not nosave:
                # 保存为视频
                if vid_path[0] != str(save_dir / f"{Path(source).stem}_detected.mp4"):
                    vid_path[0] = str(save_dir / f"{Path(source).stem}_detected.mp4")
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()
                    
                    h, w = im0.shape[:2]
                    vid_writer[0] = cv2.VideoWriter(vid_path[0], cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0)

            # 显示进度
            if processed_frames % 30 == 0 or processed_frames < 10:
                elapsed_time = time.time() - start_time_process
                if processed_frames > 0:
                    avg_time_per_frame = elapsed_time / processed_frames
                    estimated_total = min(max_frames, end_frame - start_frame)
                    remaining_time = avg_time_per_frame * (estimated_total - processed_frames)
                    print(f"📊 帧 {current_frame_num} ({processed_frames}/{estimated_total}) | "
                          f"检测数: {len(det) if len(det) else 0} | "
                          f"推理: {dt[1].dt*1000:.1f}ms | 剩余: {remaining_time:.1f}s")

        processed_frames += 1
        frame_count += 1
        
        # 显示图像（如果启用）
        if view_img:
            cv2.imshow("YOLOv5 Detection", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    
    # 释放视频写入器
    for writer in vid_writer:
        if isinstance(writer, cv2.VideoWriter):
            writer.release()

    # 计算最终统计
    for class_name in detection_summary:
        count = detection_summary[class_name]['count']
        if count > 0:
            detection_summary[class_name]['avg_conf'] = detection_summary[class_name]['total_conf'] / count

    # 保存详细统计信息
    statistics = {
        'video_info': {
            'path': str(first_video),
            'name': first_video.name,
            'total_frames': seen,
            'weights_used': weights
        },
        'detection_summary': detection_summary,
        'frame_statistics': frame_statistics,
        'performance': {
            'avg_preprocess_time': dt[0].t / seen * 1000,
            'avg_inference_time': dt[1].t / seen * 1000,
            'avg_nms_time': dt[2].t / seen * 1000,
            'total_time': sum(x.t for x in dt),
            'fps': seen / sum(x.t for x in dt)
        },
        'model_info': {
            'classes': names,
            'image_size': imgsz,
            'conf_threshold': conf_thres,
            'iou_threshold': iou_thres
        }
    }

    # 保存统计文件
    stats_file = save_dir / 'statistics' / 'detailed_results.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    # 保存简化的CSV统计
    import pandas as pd
    frame_data = []
    for frame_num, frame_info in frame_statistics.items():
        frame_data.append({
            'frame': frame_num,
            'detections': frame_info['detections_count'],
            'inference_time_ms': frame_info['inference_time']
        })
    
    df = pd.DataFrame(frame_data)
    df.to_csv(save_dir / 'statistics' / 'frame_statistics.csv', index=False)

    # 输出最终统计
    t = tuple(x.t / seen * 1E3 for x in dt)
    total_detections = sum(stats['count'] for stats in detection_summary.values())
    
    print(f"\n" + "=" * 60)
    print(f"✅ 视频检测完成!")
    print(f"📹 视频: {first_video.name}")
    print(f"🎯 权重: {weights}")
    print(f"📊 总帧数: {seen}")
    print(f"🔍 总检测数: {total_detections}")
    print(f"⏱️  平均时间: 前处理 {t[0]:.1f}ms | 推理 {t[1]:.1f}ms | NMS {t[2]:.1f}ms")
    print(f"🚀 平均FPS: {statistics['performance']['fps']:.1f}")
    print(f"\n📂 详细结果保存在: {save_dir}")
    if not nosave:
        if save_as_images:
            print(f"   ├── 检测图像: {save_dir}/images/")
        else:
            print(f"   ├── 检测视频: {vid_path[0] if vid_path[0] else '未保存'}")
    print(f"   ├── 标签文件: {save_dir}/labels/")
    print(f"   ├── 裁剪图像: {save_dir}/crops/")
    print(f"   └── 统计数据: {save_dir}/statistics/")
    
    # 显示类别统计
    print(f"\n📈 检测类别统计:")
    for class_name, stats in detection_summary.items():
        if stats['count'] > 0:
            print(f"   {class_name}: {stats['count']} 个 (平均置信度: {stats['avg_conf']:.3f})")
    
    print("=" * 60)

    return statistics, save_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/small_target_train/exp7/weights/best.pt', help='权重文件路径')
    parser.add_argument('--source', type=str, default='/home/lkx/Videos/2025_CUADC_FORWARD/', help='视频源路径')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='最大检测数量')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='推理尺寸 (像素)')
    parser.add_argument('--device', default='0', help='设备 cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存结果到 *.txt')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度到 --save-txt 标签')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的预测框')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤: --classes 0, 或 --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    parser.add_argument('--project', default='runs/detect', help='保存结果到 project/name')
    parser.add_argument('--name', default='exp', help='保存结果到 project/name')
    parser.add_argument('--exist-ok', action='store_true', help='现有的 project/name ok，不增加')
    parser.add_argument('--line-thickness', default=3, type=int, help='边界框厚度 (像素)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧率步长')
    parser.add_argument('--max-frames', type=int, default=3000, help='最大测试帧数')
    parser.add_argument('--start-time', type=str, default=None, help='开始时间 (格式: HH:MM:SS)')
    parser.add_argument('--end-time', type=str, default=None, help='结束时间 (格式: HH:MM:SS)')
    parser.add_argument('--roi-params', type=float, nargs=4, default=None, help='ROI参数: x_percent y_percent width_percent height_percent')
    parser.add_argument('--roi-mode', type=str, default='percentage', choices=['percentage', 'absolute'], help='ROI模式')
    parser.add_argument('--save-as-images', action='store_true', help='将结果保存为单独的图片')
    
    args = parser.parse_args()
    
    # 处理imgsz参数
    imgsz = args.imgsz
    if isinstance(imgsz, list):
        if len(imgsz) == 1:
            imgsz = imgsz[0]
        elif len(imgsz) == 2:
            imgsz = tuple(imgsz)
        else:
            imgsz = imgsz[0]
    
    print("🚀 YOLOv5 小目标检测 - 视频测试")
    print("=" * 60)
    print(f"🎯 权重文件: {args.weights}")
    print(f"📁 视频源: {args.source}")
    print(f"🔍 置信度阈值: {args.conf_thres}")
    print(f"📊 IOU阈值: {args.iou_thres}")
    print(f"🔢 最大检测数: {args.max_det}")
    print(f"📐 图像尺寸: {imgsz}")
    print(f"💻 设备: {args.device}")
    print(f"⚡ 半精度: {args.half}")
    print(f"🎬 视频步长: {args.vid_stride}")
    print(f"📊 最大帧数: {args.max_frames}")
    print(f"💾 保存选项: txt={args.save_txt}, conf={args.save_conf}, crop={args.save_crop}")
    print(f"⏰ 时间范围: {args.start_time} 到 {args.end_time}")
    print(f"🎯 ROI参数: {args.roi_params} ({args.roi_mode}模式)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        results, save_dir = run_single_video_test(
            weights=args.weights,
            source=args.source,
            imgsz=imgsz,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            device=args.device,
            view_img=args.view_img,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            nosave=args.nosave,
            classes=args.classes,
            agnostic_nms=args.agnostic_nms,
            augment=args.augment,
            visualize=args.visualize,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            line_thickness=args.line_thickness,
            hide_labels=args.hide_labels,
            hide_conf=args.hide_conf,
            half=args.half,
            dnn=args.dnn,
            vid_stride=args.vid_stride,
            max_frames=args.max_frames,
            start_time=args.start_time,
            end_time=args.end_time,
            roi_params=args.roi_params,
            roi_mode=args.roi_mode,
            save_as_images=args.save_as_images
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️  总处理时间: {total_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()