#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 Graphical Test Interface
Support weight file selection, test folder selection, parameter configuration and TensorRT acceleration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter.font
import threading
import os
import sys
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime
import cv2
import numpy as np

# Add project path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 向上两级到yolov5-v7目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class YOLOv5TestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 Small Target Detection Test Tool")
        self.root.geometry("900x1000")
        
        # Configure font to ensure compatibility
        self.setup_fonts()
        
        # Log related
        self.log_file = None
        self.auto_save_logs = tk.BooleanVar(value=True)
        self.log_dir = Path("logs/gui_tests")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Variable definitions
        self.weights_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.test_mode = tk.StringVar(value="video")
        
        # Parameter variables
        self.conf_thres = tk.DoubleVar(value=0.25)
        self.iou_thres = tk.DoubleVar(value=0.45)
        self.max_det = tk.IntVar(value=1000)
        self.imgsz = tk.IntVar(value=640)
        self.device = tk.StringVar(value="0")
        
        # Feature switches
        self.save_txt = tk.BooleanVar(value=True)
        self.save_conf = tk.BooleanVar(value=True)
        self.save_crop = tk.BooleanVar(value=True)
        self.view_img = tk.BooleanVar(value=False)
        self.half_precision = tk.BooleanVar(value=True)
        self.use_tensorrt = tk.BooleanVar(value=False)
        self.augment = tk.BooleanVar(value=False)
        self.agnostic_nms = tk.BooleanVar(value=False)
        
        # Advanced settings
        self.max_frames = tk.IntVar(value=300)
        self.vid_stride = tk.IntVar(value=1)
        self.line_thickness = tk.IntVar(value=3)
        
        # Video time range settings
        self.enable_time_range = tk.BooleanVar(value=False)
        self.start_time = tk.StringVar(value="00:00:00")
        self.end_time = tk.StringVar(value="00:01:00")
        self.save_as_images = tk.BooleanVar(value=False)
        
        # ROI settings
        self.enable_roi = tk.BooleanVar(value=False)
        self.roi_mode = tk.StringVar(value="percentage")  # percentage or absolute
        self.roi_x_percent = tk.DoubleVar(value=10.0)
        self.roi_y_percent = tk.DoubleVar(value=10.0)
        self.roi_width_percent = tk.DoubleVar(value=80.0)
        self.roi_height_percent = tk.DoubleVar(value=80.0)
        self.roi_x_abs = tk.IntVar(value=100)
        self.roi_y_abs = tk.IntVar(value=100)
        self.roi_width_abs = tk.IntVar(value=800)
        self.roi_height_abs = tk.IntVar(value=600)
        
        self.setup_ui()
        
        # Set default paths
        self.set_default_paths()
    
    def setup_fonts(self):
        """Setup fonts to ensure compatibility with Chinese characters"""
        # 使用最简单可靠的方案：系统默认字体
        self.default_font = None  # 使用系统默认字体
        self.title_font = ("", 14, "bold")  # 空字符串表示使用默认字体族
        self.label_font = None  # 使用系统默认字体
        
        print("使用系统默认字体以确保中文显示正常")
    
        # 不设置全局字体配置，让每个组件使用系统默认字体

    def setup_ui(self):
        """设置用户界面"""
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重 - 让左右两列都可以扩展
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # 左侧功能区
        main_frame.columnconfigure(1, weight=1)  # 右侧日志区
        main_frame.rowconfigure(0, weight=1)
        
        # 创建左侧功能区域
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(1, weight=1)
        
        # 创建右侧日志区域
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        row = 0
        
        # 标题 - 直接使用unicode编码的中文字符
        title_text = "YOLOv5 小目标检测测试工具"
        title_label = tk.Label(left_frame, text=title_text, 
                          font=self.title_font)
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # 文件选择区域
        self.create_file_selection_section(left_frame, row)
        row += 5
        
        # 测试模式选择
        self.create_test_mode_section(left_frame, row)
        row += 2
        
        # 基本参数配置
        self.create_basic_params_section(left_frame, row)
        row += 6
        
        # 功能开关
        self.create_features_section(left_frame, row)
        row += 4
        
        # 高级设置
        self.create_advanced_section(left_frame, row)
        row += 4
        
        # 控制按钮
        self.create_control_buttons(left_frame, row)
        row += 2
        
        # 日志显示区域 - 移到右侧
        self.create_log_section(right_frame, 0)
        
    def create_file_selection_section(self, parent, start_row):
        """创建文件选择区域"""
        # 权重文件选择
        tk.Label(parent, text="权重文件:", font=self.label_font).grid(row=start_row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.weights_path, width=50, font=self.default_font).grid(
            row=start_row, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(parent, text="浏览", command=self.browse_weights).grid(
            row=start_row, column=2, padx=(5, 0), pady=5)
        
        # 测试源选择
        tk.Label(parent, text="测试源:", font=self.label_font).grid(row=start_row+1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.source_path, width=50, font=self.default_font).grid(
            row=start_row+1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(parent, text="浏览", command=self.browse_source).grid(
            row=start_row+1, column=2, padx=(5, 0), pady=5)
        
        # 输出目录选择
        tk.Label(parent, text="输出目录:", font=self.label_font).grid(row=start_row+2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.output_path, width=50, font=self.default_font).grid(
            row=start_row+2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(parent, text="浏览", command=self.browse_output).grid(
            row=start_row+2, column=2, padx=(5, 0), pady=5)
        
        # TensorRT 引擎文件选择/生成
        self.tensorrt_frame = ttk.LabelFrame(parent, text="TensorRT 加速", padding="5")
        self.tensorrt_frame.grid(row=start_row+3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Checkbutton(self.tensorrt_frame, text="启用 TensorRT 加速", 
                       variable=self.use_tensorrt).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(self.tensorrt_frame, text="生成 TensorRT 引擎", 
                  command=self.generate_tensorrt_engine).grid(row=0, column=1, padx=(10, 0))
        
    def create_test_mode_section(self, parent, start_row):
        """创建测试模式选择"""
        mode_frame = ttk.LabelFrame(parent, text="测试模式", padding="5")
        mode_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Radiobutton(mode_frame, text="视频检测", variable=self.test_mode, 
                       value="video").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="图片检测", variable=self.test_mode, 
                       value="image").grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="批量检测", variable=self.test_mode, 
                       value="batch").grid(row=0, column=2, sticky=tk.W)
        
    def create_basic_params_section(self, parent, start_row):
        """创建基本参数配置"""
        params_frame = ttk.LabelFrame(parent, text="基本参数", padding="5")
        params_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 置信度阈值
        tk.Label(params_frame, text="置信度阈值:", font=self.label_font).grid(row=0, column=0, sticky=tk.W, pady=2)
        conf_scale = ttk.Scale(params_frame, from_=0.01, to=1.0, variable=self.conf_thres, 
                              orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        conf_label = ttk.Label(params_frame, text="0.25")
        conf_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        
        def update_conf_label(*args):
            conf_label.config(text=f"{self.conf_thres.get():.2f}")
        self.conf_thres.trace('w', update_conf_label)
        
        # IOU阈值
        tk.Label(params_frame, text="IOU阈值:", font=self.label_font).grid(row=1, column=0, sticky=tk.W, pady=2)
        iou_scale = ttk.Scale(params_frame, from_=0.01, to=1.0, variable=self.iou_thres, 
                             orient=tk.HORIZONTAL, length=200)
        iou_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        iou_label = ttk.Label(params_frame, text="0.45")
        iou_label.grid(row=1, column=2, sticky=tk.W, pady=2)
        
        def update_iou_label(*args):
            iou_label.config(text=f"{self.iou_thres.get():.2f}")
        self.iou_thres.trace('w', update_iou_label)
        
        # 图像尺寸
        tk.Label(params_frame, text="图像尺寸:", font=self.label_font).grid(row=2, column=0, sticky=tk.W, pady=2)
        size_combo = ttk.Combobox(params_frame, textvariable=self.imgsz, 
                             values=[320, 416, 512, 640, 832, 1280], width=10)
        size_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 设备选择
        tk.Label(params_frame, text="设备:", font=self.label_font).grid(row=3, column=0, sticky=tk.W, pady=2)
        device_combo = ttk.Combobox(params_frame, textvariable=self.device, 
                               values=["cpu", "0", "1", "0,1"], width=10)
        device_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 最大检测数
        tk.Label(params_frame, text="最大检测数:", font=self.label_font).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(params_frame, from_=1, to=10000, textvariable=self.max_det, 
               width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
    def create_features_section(self, parent, start_row):
        """创建功能开关"""
        features_frame = ttk.LabelFrame(parent, text="功能选项", padding="5")
        features_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 第一行
        ttk.Checkbutton(features_frame, text="保存标签文件", 
                       variable=self.save_txt).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(features_frame, text="保存置信度", 
                       variable=self.save_conf).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(features_frame, text="保存裁剪图像", 
                       variable=self.save_crop).grid(row=0, column=2, sticky=tk.W)
        
        # 第二行
        ttk.Checkbutton(features_frame, text="显示结果", 
                       variable=self.view_img).grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(features_frame, text="半精度推理", 
                       variable=self.half_precision).grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(features_frame, text="数据增强", 
                       variable=self.augment).grid(row=1, column=2, sticky=tk.W)
        
        # 第三行
        ttk.Checkbutton(features_frame, text="类别无关NMS", 
                       variable=self.agnostic_nms).grid(row=2, column=0, sticky=tk.W, padx=(0, 20))
        
    def create_advanced_section(self, parent, start_row):
        """创建高级设置"""
        advanced_frame = ttk.LabelFrame(parent, text="高级设置", padding="5")
        advanced_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 最大测试帧数（仅视频）
        tk.Label(advanced_frame, text="最大测试帧数:", font=self.label_font).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=1, to=10000, textvariable=self.max_frames, 
                   width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 视频步幅
        tk.Label(advanced_frame, text="视频步幅:", font=self.label_font).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=1, to=10, textvariable=self.vid_stride, 
                   width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 线条粗细
        tk.Label(advanced_frame, text="线条粗细:", font=self.label_font).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(advanced_frame, from_=1, to=10, textvariable=self.line_thickness, 
                   width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 视频时间范围
        tk.Label(advanced_frame, text="时间范围:", font=self.label_font).grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(advanced_frame, text="启用时间范围", 
                       variable=self.enable_time_range).grid(row=3, column=1, sticky=tk.W)
        
        tk.Label(advanced_frame, text="起始时间:", font=self.label_font).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.start_time, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="结束时间:", font=self.label_font).grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.end_time, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        # ROI设置
        tk.Label(advanced_frame, text="ROI设置:", font=self.label_font).grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(advanced_frame, text="启用ROI", 
                       variable=self.enable_roi).grid(row=6, column=1, sticky=tk.W)
        
        tk.Label(advanced_frame, text="模式:", font=self.label_font).grid(row=7, column=0, sticky=tk.W, pady=2)
        roi_mode_combo = ttk.Combobox(advanced_frame, textvariable=self.roi_mode, 
                                     values=["percentage", "absolute"], width=10)
        roi_mode_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)
        
        # ROI百分比设置
        tk.Label(advanced_frame, text="X坐标(%):", font=self.label_font).grid(row=8, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_x_percent, width=10).grid(row=8, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="Y坐标(%):", font=self.label_font).grid(row=9, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_y_percent, width=10).grid(row=9, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="宽度(%):", font=self.label_font).grid(row=10, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_width_percent, width=10).grid(row=10, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="高度(%):", font=self.label_font).grid(row=11, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_height_percent, width=10).grid(row=11, column=1, sticky=tk.W, padx=5, pady=2)
        
        # ROI绝对值设置
        tk.Label(advanced_frame, text="X坐标(像素):", font=self.label_font).grid(row=8, column=2, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_x_abs, width=10).grid(row=8, column=3, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="Y坐标(像素):", font=self.label_font).grid(row=9, column=2, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_y_abs, width=10).grid(row=9, column=3, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="宽度(像素):", font=self.label_font).grid(row=10, column=2, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_width_abs, width=10).grid(row=10, column=3, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(advanced_frame, text="高度(像素):", font=self.label_font).grid(row=11, column=2, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.roi_height_abs, width=10).grid(row=11, column=3, sticky=tk.W, padx=5, pady=2)
        
    def create_control_buttons(self, parent, start_row):
        """创建控制按钮"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=start_row, column=0, columnspan=3, pady=20)
        
        # 开始测试按钮
        self.start_button = ttk.Button(button_frame, text="开始测试", 
                                      command=self.start_test, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # 停止测试按钮
        self.stop_button = ttk.Button(button_frame, text="停止测试", 
                                     command=self.stop_test, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # 清除日志按钮
        ttk.Button(button_frame, text="清除日志", 
                  command=self.clear_log).grid(row=0, column=2, padx=(0, 10))
        
        # 保存配置按钮
        ttk.Button(button_frame, text="保存配置", 
                  command=self.save_config).grid(row=0, column=3, padx=(0, 10))
        
        # 加载配置按钮
        ttk.Button(button_frame, text="加载配置", 
                  command=self.load_config).grid(row=0, column=4)
        
    def create_log_section(self, parent, start_row):
        """创建日志显示区域"""
        log_frame = ttk.LabelFrame(parent, text="运行日志", padding="5")
        log_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置主框架的行权重，使日志区域可以扩展
        parent.rowconfigure(start_row, weight=1)
        
    def set_default_paths(self):
        """设置默认路径"""
        # 默认权重文件
        default_weights = Path("runs/small_target_train/exp7/weights/best.pt")
        if default_weights.exists():
            self.weights_path.set(str(default_weights))
        
        # 默认测试源
        default_source = Path("/home/lkx/Videos/2025_CUADC_FORWARD/")
        if default_source.exists():
            self.source_path.set(str(default_source))
        
        # 默认输出目录
        self.output_path.set("runs/detect/gui_test")
        
    def browse_weights(self):
        """浏览权重文件"""
        filename = filedialog.askopenfilename(
            title="选择权重文件",
            filetypes=[("权重文件", "*.pt *.engine"), ("所有文件", "*.*")],
            initialdir="runs/small_target_train"
        )
        if filename:
            self.weights_path.set(filename)
            
    def browse_source(self):
        """浏览测试源"""
        if self.test_mode.get() == "image":
            # 图片检测模式 - 选择包含图片的文件夹
            dirname = filedialog.askdirectory(title="选择包含图片的文件夹")
            if dirname:
                self.source_path.set(dirname)
                # 检查文件夹中是否有图片文件
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(Path(dirname).glob(f'*{ext}'))
                    image_files.extend(Path(dirname).glob(f'*{ext.upper()}'))
                
                if image_files:
                    self.log_message(f"📁 选择的文件夹包含 {len(image_files)} 个图片文件")
                else:
                    messagebox.showwarning("警告", "选择的文件夹中没有找到图片文件")
        else:
            # 视频和批量检测模式 - 选择文件夹
            dirname = filedialog.askdirectory(title="选择测试目录")
            if dirname:
                self.source_path.set(dirname)
                
    def browse_output(self):
        """浏览输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_path.set(dirname)
            
    def generate_tensorrt_engine(self):
        """生成TensorRT引擎"""
        if not self.weights_path.get():
            messagebox.showerror("错误", "请先选择权重文件")
            return
            
        weights_file = Path(self.weights_path.get())
        if not weights_file.exists():
            messagebox.showerror("错误", "权重文件不存在")
            return
            
        # 生成引擎文件名
        engine_file = weights_file.with_suffix('.engine')
        
        self.log_message(f"开始生成TensorRT引擎文件: {engine_file}")
        
        # 在后台线程中运行导出命令
        def export_engine():
            try:
                cmd = [
                    sys.executable, "export.py",
                    "--weights", str(weights_file),
                    "--include", "engine",
                    "--imgsz", str(self.imgsz.get()),
                    "--device", self.device.get()
                ]
                
                if self.half_precision.get():
                    cmd.append("--half")
                    
                self.log_message(f"执行命令: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, cwd=ROOT
                )
                
                for line in process.stdout:
                    self.log_message(line.strip())
                    
                process.wait()
                
                if process.returncode == 0:
                    self.log_message("TensorRT引擎生成成功!")
                    messagebox.showinfo("成功", f"TensorRT引擎已生成: {engine_file}")
                else:
                    self.log_message("TensorRT引擎生成失败!")
                    messagebox.showerror("错误", "TensorRT引擎生成失败")
                    
            except Exception as e:
                self.log_message(f"生成TensorRT引擎时发生错误: {e}")
                messagebox.showerror("错误", f"生成TensorRT引擎时发生错误: {e}")
                
        threading.Thread(target=export_engine, daemon=True).start()
        
    def start_test(self):
        """开始测试"""
        # 验证必要的参数
        if not self.weights_path.get():
            messagebox.showerror("错误", "请选择权重文件")
            return
            
        if not self.source_path.get():
            messagebox.showerror("错误", "请选择测试源")
            return
            
        weights_file = Path(self.weights_path.get())
        if not weights_file.exists():
            messagebox.showerror("错误", "权重文件不存在")
            return
            
        source_path = Path(self.source_path.get())
        if not source_path.exists():
            messagebox.showerror("错误", "测试源不存在")
            return
            
        # 禁用开始按钮，启用停止按钮
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 在后台线程中运行测试
        self.test_thread = threading.Thread(target=self.run_test, daemon=True)
        self.test_thread.start()
        
    def run_test(self):
        """运行测试"""
        try:
            self.log_message("="*60)
            self.log_message("开始YOLOv5测试")
            self.log_message(f"权重文件: {self.weights_path.get()}")
            self.log_message(f"测试源: {self.source_path.get()}")
            self.log_message(f"测试模式: {self.test_mode.get()}")
            self.log_message("="*60)
            
            # 统一使用test_videos.py脚本
            script_path = ROOT / "small_target_detection" / "scripts" / "test_videos.py"
            if not script_path.exists():
                self.log_message(f"错误: 找不到脚本文件 {script_path}")
                raise FileNotFoundError(f"脚本文件不存在: {script_path}")
            script = str(script_path)
                
            cmd = [sys.executable, script]
            
            # 添加基本参数
            cmd.extend(["--weights", self.weights_path.get()])
            cmd.extend(["--source", self.source_path.get()])
            cmd.extend(["--conf-thres", str(self.conf_thres.get())])
            cmd.extend(["--iou-thres", str(self.iou_thres.get())])
            cmd.extend(["--max-det", str(self.max_det.get())])
            cmd.extend(["--imgsz", str(self.imgsz.get())])
            cmd.extend(["--device", self.device.get()])
            
            if self.output_path.get():
                cmd.extend(["--project", self.output_path.get()])
                
            # 添加功能开关
            if self.save_txt.get():
                cmd.append("--save-txt")
            if self.save_conf.get():
                cmd.append("--save-conf")
            if self.save_crop.get():
                cmd.append("--save-crop")
            if self.view_img.get():
                cmd.append("--view-img")
            if self.half_precision.get():
                cmd.append("--half")
            if self.augment.get():
                cmd.append("--augment")
            if self.agnostic_nms.get():
                cmd.append("--agnostic-nms")
                
            # 添加高级参数
            cmd.extend(["--vid-stride", str(self.vid_stride.get())])
            cmd.extend(["--max-frames", str(self.max_frames.get())])
            cmd.extend(["--line-thickness", str(self.line_thickness.get())])
            
            # 添加时间范围参数（仅对视频有效）
            if self.enable_time_range.get() and self.test_mode.get() == "video":
                cmd.extend(["--start-time", self.start_time.get()])
                cmd.extend(["--end-time", self.end_time.get()])
                
            # 添加ROI参数
            if self.enable_roi.get():
                roi_params = []
                if self.roi_mode.get() == "percentage":
                    roi_params = [
                        self.roi_x_percent.get(),
                        self.roi_y_percent.get(),
                        self.roi_width_percent.get(),
                        self.roi_height_percent.get()
                    ]
                else:  # absolute
                    roi_params = [
                        self.roi_x_abs.get(),
                        self.roi_y_abs.get(),
                        self.roi_width_abs.get(),
                        self.roi_height_abs.get()
                    ]
                cmd.extend(["--roi-params"] + [str(p) for p in roi_params])
                cmd.extend(["--roi-mode", self.roi_mode.get()])
            
            self.log_message(f"执行命令: {' '.join(cmd)}")
            
            # 运行命令 - 确保在正确的工作目录中运行
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, cwd=ROOT
            )
            
            # 实时显示输出
            for line in self.process.stdout:
                if hasattr(self, 'process'):  # 检查进程是否被停止
                    self.log_message(line.strip())
                else:
                    break
                    
            self.process.wait()
            
            if hasattr(self, 'process') and self.process.returncode == 0:
                self.log_message("测试完成!")
                self.root.after(0, lambda: messagebox.showinfo("成功", "测试完成!"))
            elif hasattr(self, 'process'):
                self.log_message("测试失败!")
                self.root.after(0, lambda: messagebox.showerror("错误", "测试失败!"))
                
        except Exception as e:
            self.log_message(f"测试过程中发生错误: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"测试过程中发生错误: {e}"))
        finally:
            # 重新启用按钮
            self.root.after(0, self.reset_buttons)
            
    def stop_test(self):
        """停止测试"""
        if hasattr(self, 'process'):
            self.process.terminate()
            delattr(self, 'process')
            self.log_message("测试已停止")
            
        self.reset_buttons()
        
    def reset_buttons(self):
        """重置按钮状态"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
    def log_message(self, message):
        """添加日志消息"""
        # 自动保存日志到文件
        if self.auto_save_logs.get() and self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        
        def update_log():
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
            self.log_text.see(tk.END)
            
        self.root.after(0, update_log)
        
    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)
        
    def save_config(self):
        """保存配置"""
        config = {
            'weights_path': self.weights_path.get(),
            'source_path': self.source_path.get(),
            'output_path': self.output_path.get(),
            'test_mode': self.test_mode.get(),
            'conf_thres': self.conf_thres.get(),
            'iou_thres': self.iou_thres.get(),
            'max_det': self.max_det.get(),
            'imgsz': self.imgsz.get(),
            'device': self.device.get(),
            'save_txt': self.save_txt.get(),
            'save_conf': self.save_conf.get(),
            'save_crop': self.save_crop.get(),
            'view_img': self.view_img.get(),
            'half_precision': self.half_precision.get(),
            'use_tensorrt': self.use_tensorrt.get(),
            'augment': self.augment.get(),
            'agnostic_nms': self.agnostic_nms.get(),
            'max_frames': self.max_frames.get(),
            'vid_stride': self.vid_stride.get(),
            'line_thickness': self.line_thickness.get(),
            'enable_time_range': self.enable_time_range.get(),
            'start_time': self.start_time.get(),
            'end_time': self.end_time.get(),
            'save_as_images': self.save_as_images.get(),
            'enable_roi': self.enable_roi.get(),
            'roi_mode': self.roi_mode.get(),
            'roi_x_percent': self.roi_x_percent.get(),
            'roi_y_percent': self.roi_y_percent.get(),
            'roi_width_percent': self.roi_width_percent.get(),
            'roi_height_percent': self.roi_height_percent.get(),
            'roi_x_abs': self.roi_x_abs.get(),
            'roi_y_abs': self.roi_y_abs.get(),
            'roi_width_abs': self.roi_width_abs.get(),
            'roi_height_abs': self.roi_height_abs.get(),
        }
        
        filename = filedialog.asksaveasfilename(
            title="保存配置文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            defaultextension=".json"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("成功", "配置已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {e}")
                
    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            title="加载配置文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 设置变量值
                self.weights_path.set(config.get('weights_path', ''))
                self.source_path.set(config.get('source_path', ''))
                self.output_path.set(config.get('output_path', ''))
                self.test_mode.set(config.get('test_mode', 'video'))
                self.conf_thres.set(config.get('conf_thres', 0.25))
                self.iou_thres.set(config.get('iou_thres', 0.45))
                self.max_det.set(config.get('max_det', 1000))
                self.imgsz.set(config.get('imgsz', 640))
                self.device.set(config.get('device', '0'))
                self.save_txt.set(config.get('save_txt', True))
                self.save_conf.set(config.get('save_conf', True))
                self.save_crop.set(config.get('save_crop', True))
                self.view_img.set(config.get('view_img', False))
                self.half_precision.set(config.get('half_precision', True))
                self.use_tensorrt.set(config.get('use_tensorrt', False))
                self.augment.set(config.get('augment', False))
                self.agnostic_nms.set(config.get('agnostic_nms', False))
                self.max_frames.set(config.get('max_frames', 300))
                self.vid_stride.set(config.get('vid_stride', 1))
                self.line_thickness.set(config.get('line_thickness', 3))
                self.enable_time_range.set(config.get('enable_time_range', False))
                self.start_time.set(config.get('start_time', '00:00:00'))
                self.end_time.set(config.get('end_time', '00:01:00'))
                self.save_as_images.set(config.get('save_as_images', False))
                self.enable_roi.set(config.get('enable_roi', False))
                self.roi_mode.set(config.get('roi_mode', 'percentage'))
                self.roi_x_percent.set(config.get('roi_x_percent', 10.0))
                self.roi_y_percent.set(config.get('roi_y_percent', 10.0))
                self.roi_width_percent.set(config.get('roi_width_percent', 80.0))
                self.roi_height_percent.set(config.get('roi_height_percent', 80.0))
                self.roi_x_abs.set(config.get('roi_x_abs', 100))
                self.roi_y_abs.set(config.get('roi_y_abs', 100))
                self.roi_width_abs.set(config.get('roi_width_abs', 800))
                self.roi_height_abs.set(config.get('roi_height_abs', 600))
                
                messagebox.showinfo("成功", "配置已加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {e}")

def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置主题
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    
    app = YOLOv5TestGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()