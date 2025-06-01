# YOLOv5 🚀 小目标检测训练脚本
# 集成了文档中提到的所有数据增强策略

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 添加YOLOv5路径
import os
FILE = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
ROOT = FILE.parents[2]  # 向上两级到yolov5-v7目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import val as validate
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


class SmallTargetAugmentation:
    """小目标检测专用数据增强类"""
    
    def __init__(self, img_size=640, p=0.5, augment_ratio=0.3, epoch_control=True):
        self.img_size = img_size
        self.p = p
        self.augment_ratio = augment_ratio  # 控制每个epoch中应用增强的样本比例
        self.epoch_control = epoch_control   # 是否启用epoch进度控制
        self.current_epoch = 0
        self.total_epochs = 200
        
        # 基本几何变换
        self.geometric_transform = A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.25, 
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.6
            ),
            A.Transpose(p=0.1),
        ])
        
        # 颜色和亮度调整
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=40,
                val_shift_limit=25,
                p=0.6
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.ChannelShuffle(p=0.2),
        ])
        
        # 噪声和模糊处理
        self.noise_blur_transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=1),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=1),
            ], p=0.4),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=1),
                A.MotionBlur(blur_limit=9, p=1),
                A.MedianBlur(blur_limit=7, p=1),
            ], p=0.3),
        ])
        
        # 目标区域增强
        self.target_enhance = A.Compose([
            A.OneOf([
                A.Sharpen(alpha=(0.1, 0.6), lightness=(0.5, 1.0), p=1),
                A.UnsharpMask(blur_limit=(3, 7), sigma_limit=(1.0, 3.0), alpha=(0.2, 0.8), p=1),
            ], p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.3),
        ])
        
        # 遮挡和重叠模拟
        self.occlusion_transform = A.Compose([
            A.CoarseDropout(
                max_holes=12,
                max_height=64,
                max_width=64,
                min_holes=3,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.4
            ),
            A.CoarseDropout(  # 替换原来的Cutout
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
        ])
        
        # 高级变换
        self.advanced_transform = A.Compose([
            A.Perspective(scale=(0.02, 0.15), p=0.4),  # 替换PerspectiveTransform
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=8,
                p=0.3
            ),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=3,
                p=0.1
            ),
        ])
        
        # 边界框增强
        self.bbox_params = A.BboxParams(
            format='yolo',
            min_visibility=0.2,
            label_fields=['class_labels']
        )
    
    def set_epoch_info(self, current_epoch, total_epochs):
        """设置当前训练进度信息"""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
    
    def get_augmentation_probability(self):
        """根据训练进度动态调整增强概率"""
        if not self.epoch_control:
            return self.augment_ratio
        
        # 训练前期使用较少增强，后期逐渐增加
        if self.current_epoch < self.total_epochs * 0.2:  # 前20%轮次
            return self.augment_ratio * 0.5
        elif self.current_epoch < self.total_epochs * 0.6:  # 中间40%轮次
            return self.augment_ratio
        else:  # 后期40%轮次
            return self.augment_ratio * 1.2
    
    def should_apply_augmentation(self, batch_idx, batch_size):
        """决定是否对当前样本应用增强"""
        current_prob = self.get_augmentation_probability()
        
        # 确保每个batch中只有一定比例的样本被增强
        if random.random() > current_prob:
            return False
        
        return True
    
    def apply_augmentation(self, image, bboxes, class_labels):
        """应用所有增强策略"""
        # 检查是否应该应用增强
        if not self.should_apply_augmentation(0, 1):
            return image, bboxes, class_labels
        
        try:
            # 根据训练进度选择不同强度的增强
            current_prob = self.get_augmentation_probability()
            
            # 动态调整增强强度
            if current_prob < 0.2:  # 轻度增强
                augmentations = []
                if random.random() < 0.6:
                    augmentations.append(self.color_transform)
                if random.random() < 0.4:
                    augmentations.append(self.geometric_transform)
            
            elif current_prob < 0.4:  # 中度增强
                augmentations = []
                if random.random() < 0.7:
                    augmentations.append(self.color_transform)
                if random.random() < 0.6:
                    augmentations.append(self.geometric_transform)
                if random.random() < 0.3:
                    augmentations.append(self.noise_blur_transform)
            
            else:  # 强度增强
                augmentations = []
                if random.random() < 0.8:
                    augmentations.append(self.geometric_transform)
                if random.random() < 0.7:
                    augmentations.append(self.color_transform)
                if random.random() < 0.5:
                    augmentations.append(self.noise_blur_transform)
                if random.random() < 0.4:
                    augmentations.append(self.target_enhance)
                if random.random() < 0.3:
                    augmentations.append(self.occlusion_transform)
                if random.random() < 0.4:
                    augmentations.append(self.advanced_transform)
            
            # 应用选定的增强
            if augmentations:
                combined_transform = A.Compose(
                    [transform for aug in augmentations for transform in aug.transforms],
                    bbox_params=self.bbox_params
                )
                
                result = combined_transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                return result['image'], result['bboxes'], result['class_labels']
            
            return image, bboxes, class_labels
            
        except Exception as e:
            LOGGER.warning(f"数据增强失败: {e}")
            return image, bboxes, class_labels
    
    def multi_scale_augmentation(self, image, bboxes, class_labels):
        """多尺度训练增强"""
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        scale = random.choice(scales)
        
        new_size = int(self.img_size * scale)
        
        transform = A.Compose([
            A.LongestMaxSize(max_size=new_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ], bbox_params=self.bbox_params)
        
        try:
            result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return result['image'], result['bboxes'], result['class_labels']
        except:
            return image, bboxes, class_labels
    
    def mixup_augmentation(self, image1, bboxes1, labels1, image2, bboxes2, labels2, alpha=0.5):
        """图像混合增强"""
        if random.random() > 0.1:  # 10%概率应用mixup
            return image1, bboxes1, labels1
            
        lam = np.random.beta(alpha, alpha)
        
        # 确保图像尺寸一致
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 混合图像
        mixed_image = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
        
        # 合并边界框
        mixed_bboxes = list(bboxes1) + list(bboxes2)
        mixed_labels = list(labels1) + list(labels2)
        
        return mixed_image, mixed_bboxes, mixed_labels


def enhanced_create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=None,
                             pad=0.0, rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='',
                             shuffle=False, seed=0):
    """增强版数据加载器，集成小目标增强"""
    
    # 使用原有的create_dataloader，注意标准返回顺序是 (dataloader, dataset)
    dataloader, dataset = create_dataloader(
        path, imgsz, batch_size, stride, single_cls, hyp, augment, cache,
        pad, rect, rank, workers, image_weights, quad, prefix, shuffle
    )
    
    if augment and dataset is not None:
        # 为数据集添加小目标增强，设置合适的增强比例
        augment_ratio = hyp.get('augment_ratio', 0.3) if hyp else 0.3
        augmentor = SmallTargetAugmentation(
            img_size=imgsz, 
            augment_ratio=augment_ratio,
            epoch_control=True  # 启用epoch进度控制
        )
        dataset.augmentor = augmentor
        
        # 记录增强设置
        LOGGER.info(f'数据增强设置: 增强比例={augment_ratio}, epoch控制=True')
    
    # 确保返回顺序与标准create_dataloader一致：(dataloader, dataset)
    return dataloader, dataset


def apply_log_gradient_suppression(parameters, threshold, alpha=1.0):
    """应用对数梯度抑制
    
    Args:
        parameters: 模型参数
        threshold: 梯度阈值，超过此值的梯度将被对数抑制
        alpha: 控制对数抑制的强度
    """
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            # 计算参数梯度范数
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # 对每个参数应用对数抑制
            mask = p.grad.data.abs() > threshold
            if mask.any():
                # 对超过阈值的梯度应用对数抑制: sign(g) * log(1 + alpha|g|)
                p.grad.data[mask] = p.grad.data[mask].sign() * torch.log1p(alpha * p.grad.data[mask].abs())
    
    return (total_norm ** 0.5)


def small_target_train(hyp, opt, device, callbacks):
    """小目标检测训练函数"""
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    
    callbacks.run('on_pretrain_routine_start')

    # 目录设置
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # 超参数处理
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    
    # 小目标优化的超参数调整 - 针对训练稳定性优化
    stability_params = {
        'lr0': 0.0005,            # 降低初始学习率
        'warmup_epochs': 10.0,    # 增加预热轮次
        'warmup_bias_lr': 0.02,   # 降低预热偏置学习率
        'lrf': 0.005,             # 更平缓的学习率衰减
        'obj': 0.4,               # 降低目标损失权重避免不平衡
        'gradient_clip_norm': 1.0 # 更严格的梯度裁剪
    }
    
    # 只更新未明确指定的参数
    for k, v in stability_params.items():
        if k not in hyp or opt.fix_instability:
            hyp[k] = v

    # 初始化NaN检测和恢复机制（加强版）
    nan_detection = True
    nan_detection_window = 3  # 连续多少批次出现NaN时采取措施
    nan_backoff_factor = 0.5  # 学习率回退因子
    nan_recovery_batches = 0  # 恢复正常后的批次计数
    max_nan_recovery_attempts = 3 # 最大恢复尝试次数
    nan_recovery_attempts = 0  # 当前恢复尝试计数
    last_stable_state = None  # 最后一个稳定的模型状态
    max_recoverable_nan_count = 25  # 单个epoch中可恢复的最大NaN批次数
    epoch_nan_count = 0  # 当前epoch中的NaN批次计数
    global_grad_norm_clip = hyp.get('gradient_clip_norm', 1.0)  # 全局梯度裁剪值
    
    # 获取对数梯度抑制参数
    log_grad_enabled = hyp.get('log_gradient_enabled', False)  # 是否启用对数梯度抑制 
    log_grad_threshold = hyp.get('log_gradient_threshold', 5.0)  # 对数梯度抑制阈值
    log_grad_alpha = hyp.get('log_gradient_alpha', 1.0)  # 对数抑制强度
    
    # 保存初始学习率用于恢复
    initial_lr = float(hyp['lr0'])
    current_lr = initial_lr

    LOGGER.info(colorstr('训练稳定性超参数: ') + ', '.join(f'{k}={v}' for k, v in 
                {k: hyp[k] for k in stability_params.keys()}.items()))
    opt.hyp = hyp.copy()

    # 保存配置
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 日志设置
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        # 启用详细的图表绘制
        loggers.plot_results = True
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        data_dict = loggers.remote_dataset
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # 配置
    plots = not evolve and not opt.noplots  # 确保启用图表绘制
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)
    
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')

    # 模型设置
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'从 {weights} 传输 {len(csd)}/{len(model.state_dict())} 项参数')
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    # 小目标优化的锚框设置
    if hasattr(model, 'model') and hasattr(model.model[-1], 'anchors'):
        # 针对小目标的锚框优化
        small_anchors = torch.tensor([
            [[6, 8], [10, 12], [14, 18]],      # P3/8  - 小目标
            [[18, 24], [24, 32], [32, 42]],    # P4/16 - 中小目标  
            [[48, 64], [64, 86], [96, 128]]    # P5/32 - 中等目标
        ]).float().view(model.model[-1].nl, -1, 2) / 8  # 归一化
        
        # 确保锚框在正确的设备上
        model.model[-1].anchors = small_anchors.clone().to(device)
        LOGGER.info('应用小目标优化锚框')
    
    amp = check_amp(model)

    # 冻结层
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'冻结 {k}')
            v.requires_grad = False

    # 图像尺寸
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # 批次大小
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # 优化器
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # 学习率调度器
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # 恢复训练
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DDP模式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('警告 ⚠️ DP不推荐，请使用torch.distributed.run进行最佳DDP多GPU训练。')
        model = torch.nn.DataParallel(model)

    # 同步批归一化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('使用同步批归一化')

    # 数据加载器 - 使用增强版
    train_loader, dataset = enhanced_create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
        hyp=hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
        rect=opt.rect, rank=LOCAL_RANK, workers=workers,
        image_weights=opt.image_weights, quad=opt.quad,
        prefix=colorstr('train: '), shuffle=True
    )
    
    # 为数据增强器设置训练信息
    if hasattr(dataset, 'augmentor'):
        dataset.augmentor.set_epoch_info(0, epochs)
    
    # 获取标签数据，兼容不同的数据加载器类型
    try:
        if hasattr(dataset, 'labels') and dataset.labels is not None:
            # dataset.labels是一个包含多个数组的列表，每个数组对应一张图片的标签
            dataset_labels = dataset.labels
            labels = np.concatenate(dataset.labels, 0)
        elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'labels'):
            dataset_labels = train_loader.dataset.labels
            labels = np.concatenate(train_loader.dataset.labels, 0)
        else:
            # 如果无法直接获取标签，使用默认设置
            LOGGER.warning("无法获取数据集标签，使用默认设置")
            # 创建正确格式的默认标签列表和数组
            dataset_labels = [np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)]
            labels = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
            
        # 确保labels是二维数组
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)
            
        # 验证标签格式
        if labels.shape[1] < 5:
            LOGGER.warning(f"标签格式异常: shape={labels.shape}, 使用默认标签")
            dataset_labels = [np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)]
            labels = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
            
    except Exception as e:
        LOGGER.warning(f"标签处理失败: {e}")
        dataset_labels = [np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)]
        labels = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f'标签类别 {mlc} 超过 nc={nc} 在 {data}. 可能的类别标签是 0-{nc - 1}'

    # 验证数据加载器
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
            hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
            workers=workers * 2, pad=0.5, prefix=colorstr('val: ')
        )[0]

        if not resume:
            if not opt.noautoanchor:
                # 兼容不同的数据加载器类型进行锚框检查
                try:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                except Exception as e:
                    LOGGER.warning(f"自动锚框检查跳过: {e}")
            model.half().float()

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP模式
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # 模型属性调整
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    # 使用之前已经处理过的标签数据来设置类别权重
    model.class_weights = labels_to_class_weights(dataset_labels, nc).to(device) * nc
    model.names = names

    # 开始训练
    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)
    callbacks.run('on_train_start')
    
    LOGGER.info(f'图像尺寸 {imgsz} train, {imgsz} val\n'
                f'使用 {train_loader.num_workers * WORLD_SIZE} 数据加载器工作进程\n'
                f"结果保存至 {colorstr('bold', save_dir)}\n"
                f'开始小目标检测训练 {epochs} 轮次...')
    
    # 添加NaN检测计数器
    nan_count = 0

    # 训练循环
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # 更新数据增强器的epoch信息
        if hasattr(dataset, 'augmentor'):
            dataset.augmentor.set_epoch_info(epoch, epochs)
            
            # 记录当前epoch的增强概率
            current_aug_prob = dataset.augmentor.get_augmentation_probability()
            LOGGER.info(f'Epoch {epoch}: 数据增强概率 = {current_aug_prob:.3f}')

        # 更新图像权重
        if opt.image_weights:
            # 使用安全的标签访问方式
            try:
                if hasattr(dataset, 'labels') and dataset.labels is not None:
                    dataset_labels = dataset.labels
                elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'labels'):
                    dataset_labels = train_loader.dataset.labels
                else:
                    dataset_labels = labels  # 使用之前处理的标签
                
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                iw = labels_to_image_weights(dataset_labels, nc=nc, class_weights=cw)
                
                # 检查dataset是否有相关属性
                if hasattr(dataset, 'indices') and hasattr(dataset, 'n'):
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
                elif hasattr(train_loader.dataset, 'indices') and hasattr(train_loader.dataset, 'n'):
                    train_loader.dataset.indices = random.choices(range(train_loader.dataset.n), weights=iw, k=train_loader.dataset.n)
                
            except Exception as e:
                LOGGER.warning(f"图像权重更新失败，跳过: {e}")

        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # 预热
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.4, imgsz * 1.6 + gs) // gs * gs  # 扩大尺度范围
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 前向传播
            try:
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)
                    loss, loss_items = compute_loss(pred, targets.to(device))
                    if RANK != -1:
                        loss *= WORLD_SIZE
                    if opt.quad:
                        loss *= 4.
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    LOGGER.warning(f"⚠️ CUDA内存不足: {e}")
                    torch.cuda.empty_cache()  # 清理缓存
                    loss = torch.tensor([1.0], device=device)  # 使用虚拟损失值
                    loss_items = torch.tensor([0.1, 0.1, 0.1], device=device)
                else:
                    raise e
            
            # 检查损失值是否为NaN
            is_nan_loss = torch.isnan(loss).any() or torch.isinf(loss).any() or torch.isnan(loss_items).any()
            if is_nan_loss:
                nan_count += 1
                epoch_nan_count += 1
                LOGGER.warning(f'⚠️ 第{epoch}轮第{i}批次检测到NaN/Inf损失! (连续{nan_count}/{nan_detection_window})')
                
                # 检查单个epoch中的NaN批次是否过多
                if epoch_nan_count > max_recoverable_nan_count:
                    LOGGER.error(f'❌ 单个epoch中NaN批次过多 ({epoch_nan_count}>{max_recoverable_nan_count})，训练可能已不稳定')
                    if last_stable_state and nan_recovery_attempts < max_nan_recovery_attempts:
                        LOGGER.warning(f'🔄 尝试从最后稳定点恢复 (尝试 {nan_recovery_attempts+1}/{max_nan_recovery_attempts})')
                        # 恢复到最后一个稳定状态
                        model.load_state_dict(last_stable_state['model_state'])
                        optimizer.load_state_dict(last_stable_state['optimizer_state'])
                        if ema and last_stable_state['ema_state']:
                            ema.ema.load_state_dict(last_stable_state['ema_state'])
                            
                        # 继续降低学习率
                        current_lr = current_lr * 0.3  # 更激进地降低学习率
                        for j, param_group in enumerate(optimizer.param_groups):
                            param_group['lr'] = current_lr if j == 0 else current_lr * 0.1
                            
                        LOGGER.warning(f'🔄 将学习率降低到 {current_lr:.8f}')
                        nan_count = 0
                        epoch_nan_count = 0
                        nan_recovery_attempts += 1
                        nan_recovery_batches = 0
                        LOGGER.warning(f'🔄 已恢复到批次 {last_stable_state["batch"]}')
                        continue
                    else:
                        LOGGER.error('❌ 无可用的稳定状态或已达到最大恢复尝试次数，训练无法继续')
                        return False
                
                # 连续NaN检测
                if nan_count >= nan_detection_window and nan_detection:
                    # 实施强化恢复策略
                    new_lr = max(current_lr * nan_backoff_factor, initial_lr * 0.01)  # 防止学习率过低
                    LOGGER.warning(f'⚠️ 连续检测到NaN! 将学习率从{current_lr:.8f}降低到{new_lr:.8f}')
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    current_lr = new_lr
                    nan_count = 0
                    nan_recovery_batches = 0
                    
                    # 如果损失是NaN，跳过当前批次
                    LOGGER.warning('跳过当前批次，尝试继续训练...')
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 重置批次归一化层统计信息
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.reset_running_stats()
                    
                    continue
                else:
                    # 即使不满足连续NaN检测窗口，也跳过当前批次
                    LOGGER.warning('跳过当前批次，继续处理下一批次...')
                    continue
            else:
                # 非NaN批次，增加恢复计数
                nan_recovery_batches += 1
                nan_count = 0  # 重置连续NaN计数

            # 反向传播
            scaler.scale(loss).backward()

            # 优化步骤
            if ni - last_opt_step >= accumulate:
                # 创建标志变量跟踪是否已进行unscale操作
                gradients_unscaled = False
                skip_optimizer_step = False
                
                # 应用梯度处理
                if log_grad_enabled or global_grad_norm_clip > 0:
                    # 对FP32梯度取消缩放 (只做一次)
                    scaler.unscale_(optimizer)
                    gradients_unscaled = True
                    
                    # 先应用对数梯度抑制(如果启用)
                    if log_grad_enabled:
                        avg_norm = apply_log_gradient_suppression(
                            model.parameters(),
                            threshold=log_grad_threshold,
                            alpha=log_grad_alpha
                        )
                        if i % 100 == 0:  # 每100批次记录一次
                            LOGGER.info(f"应用对数梯度抑制: 平均梯度范数 {avg_norm:.4f}")
                    
                    # 再应用常规梯度裁剪(如果设置了裁剪范数)
                    if global_grad_norm_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=global_grad_norm_clip)
    
                    # 添加梯度值监控
                    if i % 50 == 0:  # 每50批次检查一次
                        grad_stats = []
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_mean = param.grad.abs().mean().item()
                                grad_stats.append(grad_mean)
                                
                                # 详细输出大梯度值
                                if grad_mean > 1.0:
                                    LOGGER.debug(f"大梯度 {name}: {grad_mean:.4f}")
            
                        if grad_stats:
                            avg_grad = sum(grad_stats) / len(grad_stats)
                            if avg_grad > 1.0:  # 梯度过大警告阈值
                                LOGGER.warning(f"⚠️ 检测到大梯度值: {avg_grad:.4f}，可能导致不稳定")
        
                    # 检查裁剪后的梯度是否仍然有NaN
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
        
                    if has_nan_grad:
                        LOGGER.warning("⚠️ 梯度中存在NaN，跳过优化步骤")
                        skip_optimizer_step = True
    
                # 执行优化步骤 (如果没有NaN)
                if not skip_optimizer_step:
                    # 如果尚未unscale，现在进行
                    if not gradients_unscaled:
                        scaler.unscale_(optimizer)
                    
                    # 执行优化器步骤
                    scaler.step(optimizer)
                    
                    # 如果是EMA模型，更新EMA
                    if ema:
                        ema.update(model)
                    
                    # 更新最后优化步骤计数
                    last_opt_step = ni
                
                # 无论有没有进行优化步骤，都要更新scaler和清零梯度
                scaler.update()
                optimizer.zero_grad()

            # 日志记录
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                   (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                
                # 监控NaN并记录
                if torch.isnan(mloss).any():
                    LOGGER.warning(f'警告: 平均损失值中存在NaN! Epoch {epoch}, 批次 {i}')
                
                if callbacks.stop_training:
                    return

        # 学习率调度
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        if RANK in {-1, 0}:
            # 验证
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            
            if not noval or final_epoch:
                results, maps, _ = validate.run(
                    data_dict, batch_size=batch_size // WORLD_SIZE * 2, imgsz=imgsz,
                    half=amp, model=ema.ema, single_cls=single_cls, dataloader=val_loader,
                    save_dir=save_dir, plots=False, callbacks=callbacks, compute_loss=compute_loss
                )

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi
            
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # 保存模型
            if (not nosave) or (final_epoch and not evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,
                    'date': datetime.now().isoformat()
                }

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # 早停
        if RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break

    # 训练结束
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} 轮次完成，耗时 {(time.time() - t0) / 3600:.3f} 小时.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\n验证 {f}...')
                    
                    # 清理GPU缓存，避免内存不足
                    torch.cuda.empty_cache()
                    
                    # 使用CPU进行最终验证，避免GPU内存问题
                    try:
                        # 首先尝试在GPU上验证
                        results, _, _ = validate.run(
                            data_dict, batch_size=batch_size // WORLD_SIZE * 2, imgsz=imgsz,
                            model=attempt_load(f, device).half(), iou_thres=0.65 if is_coco else 0.60,
                            single_cls=single_cls, dataloader=val_loader, save_dir=save_dir,
                            save_json=is_coco, verbose=True, plots=plots, callbacks=callbacks,
                            compute_loss=compute_loss
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "out of memory" in str(e).lower():
                            LOGGER.warning(f'GPU验证失败 ({e})，切换到CPU验证...')
                            torch.cuda.empty_cache()
                            
                            # 使用CPU进行验证
                            cpu_device = torch.device('cpu')
                            results, _, _ = validate.run(
                                data_dict, batch_size=1, imgsz=imgsz,  # 使用更小的批次大小
                                model=attempt_load(f, cpu_device), iou_thres=0.65 if is_coco else 0.60,
                                single_cls=single_cls, dataloader=None, save_dir=save_dir,
                                save_json=is_coco, verbose=True, plots=plots, callbacks=callbacks,
                                compute_loss=None
                            )
                        else:
                            raise e
                    
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default=ROOT / '/home/lkx/Documents/1/2025_CUADC_Front_Camera/dataset.yaml', help='数据集配置文件')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.small-target-conservative.yaml', help='超参数文件')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮次')  # 减少默认轮次
    parser.add_argument('--batch-size', type=int, default=12, help='批次大小')  # 降低默认批次大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--nosave', action='store_true', help='不保存模型检查点')  # 修改为默认保存
    parser.add_argument('--noval', action='store_true', help='不进行验证')  # 修改为默认进行验证
    parser.add_argument('--noautoanchor', action='store_true', help='禁用自动锚框')
    parser.add_argument('--noplots', action='store_true', help='不保存图表文件')  # 修改为默认保存图表
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='进化超参数')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='图像缓存')
    parser.add_argument('--image-weights', action='store_true', help='使用加权图像选择')  # 启用图像权重
    parser.add_argument('--device', default='0', help='cuda设备')
    parser.add_argument('--multi-scale', action='store_true', default=False, help='多尺度训练')  # 默认禁用多尺度减少内存使用
    parser.add_argument('--single-cls', action='store_true', help='单类训练')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='优化器')  # 改为AdamW
    parser.add_argument('--sync-bn', action='store_true', help='同步BN')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器工作进程数')  # 减少工作进程数
    parser.add_argument('--project', default=ROOT / 'runs/small_target_train', help='保存路径')
    parser.add_argument('--name', default='exp', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖')
    parser.add_argument('--quad', action='store_true', help='四边形数据加载器')
    parser.add_argument('--cos-lr', action='store_true', default=True, help='余弦学习率')  # 默认启用余弦学习率
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑')  # 增加标签平滑
    parser.add_argument('--patience', type=int, default=50, help='早停耐心度')  # 降低早停耐心度
    parser.add_argument('--freeze', nargs='+', type=int, default=[5], help='冻结层')
    parser.add_argument('--save-period', type=int, default=10, help='保存周期')  # 每10轮保存一次
    parser.add_argument('--seed', type=int, default=0, help='全局种子')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP参数')
    parser.add_argument('--entity', default=None, help='实体')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='上传数据集')
    parser.add_argument('--bbox_interval', type=int, default=1, help='边界框日志间隔')  # 每轮记录边界框
    parser.add_argument('--artifact_alias', type=str, default='latest', help='数据集版本')
    parser.add_argument('--augment-ratio', type=float, default=0.2, help='数据增强样本比例 (0.0-1.0)')
    parser.add_argument('--augment-schedule', type=str, default='progressive', 
                       choices=['fixed', 'progressive', 'adaptive'], help='增强调度策略')
    parser.add_argument('--fix-instability', action='store_true', help='应用训练稳定性修复')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """主函数"""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_requirements()

    # 恢复训练处理
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'
        opt_data = opt.data
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)
        opt.cfg, opt.weights, opt.resume = '', str(last), True
        if is_url(opt_data):
            opt.data = check_file(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        assert len(opt.cfg) or len(opt.weights), '必须指定 --cfg 或 --weights'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/small_target_train'):
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP模式
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = '与YOLOv5多GPU DDP训练不兼容'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'自动批次大小 {msg}'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} 必须是WORLD_SIZE的倍数'
        assert torch.cuda.device_count() > LOCAL_RANK, 'CUDA设备不足'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # 训练
    if not opt.evolve:
        small_target_train(opt.hyp, opt, device, callbacks)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
