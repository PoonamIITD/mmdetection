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

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='images/train')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='images/test')))

test_dataloader = val_dataloader

# val_evaluator = dict(ann_file=data_root + 'annotations/instances_test2017.json')
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',
    classwise=True
)
test_evaluator = val_evaluator

max_epoch = 12

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

auto_scale_lr = dict(base_batch_size=16)

