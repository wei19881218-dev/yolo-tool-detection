# YOLO-World实验室工具检测项目

本项目基于YOLO-World实现实验室常见工具检测。项目在官方YOLO-World代码基础上，加入了面向实验室工具数据集的类别文本、微调配置和图片/视频推理脚本，可用于识别螺丝刀、面包板、内六角扳手、钳子、杜邦束、扳手和电烙铁等目标。

## 1.代码来源

YOLO-World原始代码来自官方开源仓库：

```bash
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```

官方仓库提供了YOLO-World的PyTorch实现、预训练权重、预训练配置和微调配置。本项目保留官方代码的基本目录结构，并在此基础上增加实验室工具检测相关文件。

## 2.项目目录结构

建议项目保持如下结构：

```text
YOLO-World/
├── assets/                         # 官方项目图片资源
├── configs/                        # 官方配置文件与自定义微调配置
│   ├── pretrain/                   # 官方预训练配置
│   ├── finetune_coco/              # 官方COCO微调配置
│   └── custom/
│       └── lab_tools_finetune_7_2_1.py
├── data/
│   ├── texts/
│   │   ├── coco_class_texts.json
│   │   └── lab_tools_class_texts.json
│   └── lab_tools/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── annotations/
│           ├── train.json
│           ├── val.json
│           └── test.json
├── demo/
│   ├── image_demo.py
│   ├── video_demo.py
│   ├── lab_tools_single_image_demo.py
│   ├── lab_tools_finetuned_single_image_demo.py
│   └── lab_tools_video_demo.py
├── pretrained_models/              # 放置官方预训练权重，不建议上传到GitHub
├── work_dirs/                      # 训练输出目录，不建议上传到GitHub
├── outputs/                        # 推理结果输出目录
├── tools/
│   ├── train.py
│   └── test.py
└── yolo_world/                     # YOLO-World核心代码
```

## 3.需要新增或重点保留的文件

从官方YOLO-World代码开始复现时，需要额外加入以下文件。

### 3.1类别文本文件

文件位置：

```text
data/texts/lab_tools_class_texts.json
```

内容如下：

```json
[
  ["screwdriver"],
  ["breadboard"],
  ["hex_key"],
  ["plier"],
  ["jumper_wire_bundle"],
  ["wrench"],
  ["soldering_iron"]
]
```

该文件用于给YOLO-World提供类别文本提示。类别顺序需要与数据集标注文件和配置文件中的类别顺序保持一致。

### 3.2微调配置文件

文件位置：

```text
configs/custom/lab_tools_finetune_7_2_1.py
```

该文件可以由官方COCO微调配置复制后修改得到，例如参考：

```text
configs/finetune_coco/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco.py
```

主要需要修改以下内容：

```python
lab_tool_classes = (
    'screwdriver',
    'breadboard',
    'hex_key',
    'plier',
    'jumper_wire_bundle',
    'wrench',
    'soldering_iron',
)

num_classes = 7
num_training_classes = 7
max_epochs = 50
close_mosaic_epochs = 10
train_batch_size_per_gpu = 4

load_from = 'pretrained_models/your_pretrained_yoloworld_weight.pth'
```

训练集配置示例：

```python
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=lab_tool_classes),
        data_root='data/lab_tools',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/lab_tools_class_texts.json',
    pipeline=train_pipeline)
```

验证集配置示例：

```python
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=lab_tool_classes),
        data_root='data/lab_tools',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/lab_tools_class_texts.json',
    pipeline=test_pipeline)

val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/lab_tools/annotations/val.json',
    metric='bbox')
```

如果显存不足，可以把`train_batch_size_per_gpu`调小，例如改为`2`或`1`。如果使用单张RTX4090，通常可以从`batch_size=4`或`8`开始尝试。

### 3.3单张图片推理脚本

文件位置：

```text
demo/lab_tools_finetuned_single_image_demo.py
```

该脚本用于加载微调后的权重，对单张图片进行检测，并输出带检测框的结果图。

### 3.4视频推理脚本

文件位置：

```text
demo/lab_tools_video_demo.py
```

该脚本用于加载微调后的权重，对视频或摄像头画面进行检测。脚本支持设置输出视频时长、输出帧率和置信度阈值。

### 3.5零样本检测脚本

文件位置：

```text
demo/lab_tools_single_image_demo.py
```

该脚本用于使用未微调的YOLO-World预训练模型进行零样本检测。它通过文本提示直接检测实验室工具类别，可用于和微调后的模型效果进行对比。

## 4.环境配置

建议使用Python3.10环境。

```bash
conda create -n yoloworld python=3.10 -y
conda activate yoloworld
```

安装PyTorch。下面命令以CUDA11.8为例，具体版本可根据服务器CUDA版本调整：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

安装MMCV、MMDetection、MMYOLO等依赖：

```bash
pip install -U openmim
mim install "mmcv==2.0.0"
pip install -r requirements/basic_requirements.txt
pip install -r requirements/demo_requirements.txt
pip install -e .
```

安装完成后，可以用下面命令检查YOLO-World是否能够被正常导入：

```bash
python -c "import torch; import yolo_world; print(torch.__version__)"
```

## 5.数据集准备

本项目使用COCO格式标注。建议将数据集整理为：

```text
data/lab_tools/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

COCO标注文件中需要包含`images`、`annotations`和`categories`字段。`categories`中的类别名称建议保持为：

```text
screwdriver
breadboard
hex_key
plier
jumper_wire_bundle
wrench
soldering_iron
```

训练、验证和测试集可以按照7:2:1划分。划分完成后，需要保证图片路径和JSON中的`file_name`能够对应，否则训练时会出现找不到图片的问题。

## 6.下载预训练权重

从YOLO-World官方仓库的Model Card中下载对应预训练权重，并放入：

```text
pretrained_models/
```

例如：

```text
pretrained_models/your_pretrained_yoloworld_weight.pth
```

然后在`configs/custom/lab_tools_finetune_7_2_1.py`中修改：

```python
load_from = 'pretrained_models/your_pretrained_yoloworld_weight.pth'
```

模型权重文件通常较大，不建议直接上传到GitHub。如果需要公开项目，可以在README中说明权重下载方式，或者将权重放到网盘、HuggingFace或GitHubRelease中。

## 7.开始微调

单卡训练可以使用：

```bash
python tools/train.py configs/custom/lab_tools_finetune_7_2_1.py --amp
```

也可以使用官方分布式训练脚本，单卡时将GPU数量设为`1`：

```bash
bash tools/dist_train.sh configs/custom/lab_tools_finetune_7_2_1.py 1 --amp
```

训练输出会保存在：

```text
work_dirs/lab_tools_finetune_7_2_1/
```

常见的输出文件包括：

```text
best_coco_bbox_mAP_epoch_xx.pth
last_checkpoint
*.log
*.json
```

其中`best_coco_bbox_mAP_epoch_xx.pth`是验证集mAP最高时保存的模型权重，后续推理建议使用这个权重。

## 8.模型测试与推理

### 8.1测试验证集指标

```bash
python tools/test.py configs/custom/lab_tools_finetune_7_2_1.py work_dirs/lab_tools_finetune_7_2_1/best_coco_bbox_mAP_epoch_45.pth
```

如果最佳权重文件名不同，需要替换为实际生成的`.pth`文件。

### 8.2微调模型单张图片检测

```bash
python demo/lab_tools_finetuned_single_image_demo.py \
  configs/custom/lab_tools_finetune_7_2_1.py \
  work_dirs/lab_tools_finetune_7_2_1/best_coco_bbox_mAP_epoch_45.pth \
  test_tool.jpg \
  --score-thr 0.25 \
  --out outputs/finetuned_lab_tools_vis.jpg
```

### 8.3视频检测

```bash
python demo/lab_tools_video_demo.py \
  configs/custom/lab_tools_finetune_7_2_1.py \
  work_dirs/lab_tools_finetune_7_2_1/best_coco_bbox_mAP_epoch_45.pth \
  --source input.mp4 \
  --out outputs/lab_tools_result.mp4 \
  --score-thr 0.25 \
  --min-seconds 25 \
  --fps-out 25
```

如果需要调用摄像头，可以将`--source input.mp4`改为：

```bash
--source 0
```

### 8.4未微调模型零样本检测

```bash
python demo/lab_tools_single_image_demo.py \
  configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
  pretrained_models/your_pretrained_yoloworld_weight.pth \
  test_tool.jpg \
  --score-thr 0.20 \
  --out outputs/original_yoloworld_lab_tools.jpg
```

该命令用于观察YOLO-World原始模型在实验室工具类别上的零样本检测效果。与微调模型结果对比，可以体现微调对小数据集和特定场景的提升作用。

## 9.常见问题

### 9.1类别数量不匹配

如果出现类别数量或检测头维度不匹配，需要检查：

```python
num_classes = 7
num_training_classes = 7
cfg.model.num_test_classes = 7
```

同时确认`lab_tools_class_texts.json`、配置文件中的`lab_tool_classes`、COCO标注文件中的`categories`顺序一致。

### 9.2找不到图片

如果训练时报错找不到图片，通常是`data_root`、`ann_file`或`data_prefix`路径不一致。需要检查配置文件中：

```python
data_root='data/lab_tools'
ann_file='annotations/train.json'
data_prefix=dict(img='images/train/')
```

最终拼接出来的路径应该能正确指向训练图片。

### 9.3显存不足

可以优先减小：

```python
train_batch_size_per_gpu = 2
```

也可以降低输入分辨率、关闭部分数据增强，或使用`--amp`进行混合精度训练。

### 9.4GitHub无法上传权重

GitHub普通仓库不适合直接上传较大的`.pth`、`.pt`或`.ckpt`文件。建议在`.gitignore`中忽略：

```text
pretrained_models/
work_dirs/
outputs/
*.pth
*.pt
*.ckpt
```

如果需要分享权重，可以使用网盘、HuggingFace或GitHubRelease，并在README中说明下载位置。

## 10.项目说明

本项目主要完成了以下工作：

1.基于YOLO-World官方代码搭建开放词汇目标检测环境；
2.构建实验室工具检测数据集，并整理为COCO格式；
3.添加实验室工具类别文本提示文件；
4.编写自定义微调配置文件，对YOLO-World进行实验室工具场景微调；
5.编写单张图片和视频检测脚本，实现微调模型的可视化推理；
6.对比原始YOLO-World零样本检测和微调后检测效果，验证微调对特定场景识别性能的提升。
