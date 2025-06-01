# update_labels_class.py - 标签类别更新脚本

**作者**: 2022级-飞行器设计与工程-雒凯星(多旋翼项目组)  
**更新时间**: 2025年5月28日  
**脚本路径：** `small_target_detection/scripts/update_labels_class.py`

## 目录
- [脚本概述](#脚本概述)
- [功能特性](#功能特性)
- [参数说明](#参数说明)
- [使用方法](#使用方法)
- [输出文件说明](#输出文件说明)
- [注意事项](#注意事项)
- [常见问题](#常见问题)

## 脚本概述

`update_labels_class.py` 是一个用于批量修改YOLO格式标签文件中类别ID的工具脚本。该脚本支持将指定的类别ID统一修改为新的类别ID，并可选择性地删除特定类别。

## 功能特性

- ✅ 批量处理YOLO格式标签文件
- ✅ 类别ID映射和重新编号
- ✅ 选择性删除指定类别
- ✅ 支持备份原始文件
- ✅ 详细的处理统计信息
- ✅ 多种输出格式支持

## 参数说明

### 基础参数

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--input-dir` | str | 必填 | 输入标签文件目录路径 | 有效的目录路径 |
| `--output-dir` | str | `input-dir + '_updated'` | 输出标签文件目录路径 | 有效的目录路径 |
| `--class-mapping` | dict | `{}` | 类别映射字典 | `{old_id: new_id}` 格式 |
| `--delete-classes` | list | `[]` | 要删除的类别ID列表 | 整数列表 |

### 高级参数

| 参数名 | 类型 | 默认值 | 说明 | 可取值 |
|--------|------|--------|------|--------|
| `--backup` | bool | `True` | 是否备份原始文件 | `True/False` |
| `--backup-suffix` | str | `'.bak'` | 备份文件后缀 | 字符串 |
| `--log-level` | str | `'INFO'` | 日志级别 | `DEBUG/INFO/WARNING/ERROR` |
| `--dry-run` | bool | `False` | 仅预览不实际修改 | `True/False` |

## 使用方法

### 1. 基本用法 - 类别映射

```bash
# 将类别0映射为类别1，类别2映射为类别0
python update_labels_class.py \
    --input-dir /path/to/labels \
    --output-dir /path/to/updated_labels \
    --class-mapping '{"0": 1, "2": 0}'
```

### 2. 删除特定类别

```bash
# 删除类别3和类别4的所有标注
python update_labels_class.py \
    --input-dir /path/to/labels \
    --delete-classes 3 4
```

### 3. 组合操作

```bash
# 同时进行类别映射和删除
python update_labels_class.py \
    --input-dir /path/to/labels \
    --output-dir /path/to/processed_labels \
    --class-mapping '{"0": 1, "1": 0}' \
    --delete-classes 2 3 \
    --backup
```

### 4. 预览模式

```bash
# 预览修改效果，不实际修改文件
python update_labels_class.py \
    --input-dir /path/to/labels \
    --class-mapping '{"0": 1}' \
    --dry-run
```

## 输出文件说明

### 1. 更新后的标签文件
- **位置**: `output-dir/` 目录下
- **格式**: YOLO格式 `.txt` 文件
- **内容**: 类别ID已按映射规则更新的标注数据

### 2. 备份文件（可选）
- **位置**: `input-dir/` 目录下
- **格式**: 原始文件名 + 备份后缀
- **示例**: `image1.txt.bak`

### 3. 处理日志
- **文件**: `processing.log`
- **内容**: 详细的处理过程和统计信息

### 4. 统计报告
- **文件**: `class_update_report.json`
- **内容**: 处理统计信息
```json
{
    "total_files": 100,
    "processed_files": 95,
    "skipped_files": 5,
    "class_changes": {
        "0->1": 150,
        "2->0": 80
    },
    "deleted_annotations": {
        "class_3": 25,
        "class_4": 10
    }
}
```

## 注意事项

### ⚠️ 重要提醒

1. **备份重要性**: 强烈建议在修改前备份原始标签文件
2. **路径检查**: 确保输入目录存在且包含 `.txt` 格式的标签文件
3. **权限要求**: 确保对输入和输出目录有读写权限
4. **类别一致性**: 修改后的类别ID应与训练配置文件保持一致

### 🔧 修改位置

要修改脚本的默认行为，可以编辑以下位置：

```python
# 文件: update_labels_class.py
# 修改默认参数
DEFAULT_CONFIG = {
    'backup': True,          # 第45行附近
    'backup_suffix': '.bak', # 第46行附近
    'log_level': 'INFO'      # 第47行附近
}
```

### 📝 配置文件支持

可以创建配置文件 `class_mapping_config.json`:

```json
{
    "class_mapping": {
        "0": 1,
        "1": 0,
        "2": 2
    },
    "delete_classes": [3, 4],
    "backup": true,
    "backup_suffix": ".original"
}
```

使用配置文件：
```bash
python update_labels_class.py \
    --input-dir /path/to/labels \
    --config class_mapping_config.json
```

## 常见问题

### Q1: 如何恢复误操作的修改？
**A**: 如果启用了备份（默认），可以从 `.bak` 文件恢复：
```bash
# 批量恢复备份文件
for file in *.txt.bak; do
    mv "$file" "${file%.bak}"
done
```

### Q2: 处理大量文件时速度较慢怎么办？
**A**: 可以使用多进程版本或分批处理：
```bash
# 分批处理
python update_labels_class.py \
    --input-dir /path/to/labels/batch1 \
    --batch-size 1000
```

### Q3: 如何验证修改结果？
**A**: 使用验证脚本检查类别分布：
```bash
python validate_labels.py \
    --labels-dir /path/to/updated_labels \
    --report-classes
```

### Q4: 支持哪些标签格式？
**A**: 目前仅支持YOLO格式：
```
class_id x_center y_center width height [confidence]
```

### Q5: 如何处理包含置信度的标签？
**A**: 脚本自动检测和保留置信度信息，无需额外配置。

---

**最后更新**: 2024年12月
**维护者**: YOLOv5 小目标检测项目组