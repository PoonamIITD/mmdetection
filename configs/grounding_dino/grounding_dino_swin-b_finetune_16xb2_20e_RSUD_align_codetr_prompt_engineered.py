_base_ = [
    './grounding_dino_swin-b_finetune_16xb2_1x_coco.py',
]

data_root = '../dataset/RSUD_dataset/'
# class_name_eval = ('person','rickshaw','rickshaw van','auto rickshaw','truck','pickup truck','private car','motorcycle','bicycle','bus','micro bus','covered van','human hauler', )
# Engineered prompts per class
class_name = ('A person is a living being with a complex physical form, including a head, torso, limbs, and varied appearance based on ethnicity and individual traits',
    'A rickshaw is a human-powered or motorized vehicle with a simple frame, seating, and often two or three wheels',
    'A rickshaw van is a motorized three-wheeled vehicle with an enclosed cabin for passengers or goods, and typically a driver upfront',
    'An auto rickshaw is a compact, three-wheeled motorized vehicle with a cabin for passengers, a driver upfront, and a rear engine',
    'A truck is a large, motorized vehicle with a driver’s cabin, cargo area, wheels, and often a distinct front grille',
    'A pickup truck is a smaller motorized vehicle with a driver’s cabin and an open cargo bed in the rear',
    'A private car is a four-wheeled motor vehicle designed for personal transportation, typically with seating for passengers and an enclosed cabin',
    'A motorcycle is a two-wheeled motor vehicle with a seat for a rider and often a pillion seat for a passenger',
    'A bicycle is a human-powered vehicle with two wheels, pedals, a frame, handlebars, and a seat for a rider',
    'A bus is a large motorized vehicle with a passenger cabin, typically featuring multiple seats, windows, and a distinctive elongated shape',
    'A micro bus is a smaller motorized vehicle, similar to a standard bus but more compact with seating for fewer passengers',
    'A covered van is a motorized vehicle with a closed cargo area, often used for transporting goods, and may have a driver’s cabin upfront',
    'A human hauler is a motorized vehicle designed for transporting passengers, similar to an auto rickshaw or tuk-tuk, with a cabin and driver upfront',
)

num_classes = len(class_name)

import random
# Generate a random palette with unique colors
def generate_palette(num_classes):
    random.seed(42)  # For reproducibility
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

pallete = generate_palette(num_classes)
metainfo = dict(classes=class_name, palette = pallete)
# metainfo_eval = dict(classes=class_name_eval, pallete=pallete)
print("DEBUG: Number of classes:", len(metainfo['classes']))
# metainfo = dict(
#     classes=class_name,
#     text=class_prompts  # this is your engineered prompts
# )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# prompt_text = ', '.join([class_prompts[c] for c in class_name])
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
# metainfo = dict(classes=class_name)

# Dataloaders setup
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017_update.json',  # Adjust path
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
        ann_file='annotations/instances_val2017_update.json',  # Adjust path
        data_prefix=dict(img='images/val'),  # Keep original test folder (not resized manually
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader  # Test data will use the same settings as validation

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_update.json')
test_evaluator = val_evaluator  # Use same evaluator for testing
max_epoch = 21

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

auto_scale_lr = dict(base_batch_size=16)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
        'language_model': dict(lr_mult=0.1)
    }))