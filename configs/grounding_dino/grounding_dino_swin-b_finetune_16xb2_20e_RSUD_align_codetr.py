_base_ = [
    './grounding_dino_swin-b_finetune_16xb2_1x_coco.py',
]

data_root = '../dataset/RSUD_dataset/'
class_name = ('person','rickshaw','rickshaw van','auto rickshaw','truck','pickup truck','private car','motorcycle','bicycle','bus','micro bus','covered van','human hauler', )
num_classes = len(class_name)

import random
# Generate a random palette with unique colors
def generate_palette(num_classes):
    random.seed(42)  # For reproducibility
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

pallete = generate_palette(num_classes)
metainfo = dict(classes=class_name, palette = pallete)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2400), (512, 2400), (544, 2400), (576, 2400),
                               (608, 2400), (640, 2400), (672, 2400), (704, 2400),
                               (736, 2400), (768, 2400), (800, 2400), (832, 2400),
                               (864, 2400), (896, 2400), (928, 2400), (960, 2400),
                               (992, 2400), (1024, 2400), (1056, 2400), (1088, 2400),
                               (1120, 2400), (1152, 2400), (1184, 2400), (1216, 2400),
                               (1248, 2400), (1280, 2400), (1312, 2400), (1344, 2400), 
                               (1376, 2400), (1408, 2400), (1440, 2400), (1472, 2400),
                               (1504, 2400), (1536, 2400)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                   # follow the original impl
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2400), (512, 2400), (544, 2400), (576, 2400),
                               (608, 2400), (640, 2400), (672, 2400), (704, 2400),
                               (736, 2400), (768, 2400), (800, 2400), (832, 2400),
                               (864, 2400), (896, 2400), (928, 2400), (960, 2400),
                               (992, 2400), (1024, 2400), (1056, 2400), (1088, 2400),
                               (1120, 2400), (1152, 2400), (1184, 2400), (1216, 2400),
                               (1248, 2400), (1280, 2400), (1312, 2400), (1344, 2400), 
                               (1376, 2400), (1408, 2400), (1440, 2400), (1472, 2400),
                               (1504, 2400), (1536, 2400)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',  # Rescaling the image
        scale=(2048, 1280),  # Resizing to the specified scale (you can modify this scale as needed)
        keep_ratio=True,  # Keep the aspect ratio intact
        backend='pillow'),  # Resizing using pillow backend
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

# Data root and metainfo (adjust paths to your setup)
data_root = '../dataset/RSUD_dataset/'  
metainfo = dict(classes=class_name)

# Dataloaders setup
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',  # Adjust path
        data_prefix=dict(img='images/train'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',  # Adjust path
        data_prefix=dict(img='images/test'),  # Keep original test folder (not resized manually
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader  # Test data will use the same settings as validation

val_evaluator = dict(ann_file=data_root + 'annotations/instances_test2017.json')
test_evaluator = val_evaluator  # Use same evaluator for testing
max_epoch = 21

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

auto_scale_lr = dict(base_batch_size=16)

