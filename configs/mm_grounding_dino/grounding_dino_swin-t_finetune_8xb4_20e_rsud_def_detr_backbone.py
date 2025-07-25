_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

data_root = '../dataset/RSUD_dataset/'
class_name = ('person','rickshaw','rickshaw van','auto rickshaw','truck','pickup truck','private car','motorcycle','bicycle','bus','micro bus','covered van','human hauler', )
num_classes = len(class_name)

import random
# Generate a random palette with unique colors
def generate_palette(num_classes):
    random.seed(42)  # For reproducibility
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

pallete = generate_palette(num_classes)
metainfo = dict(classes=class_name, palette=pallete)

model = dict(
    type = 'GroundingDINO',
    def_detr=dict(
        type='DeformableDETR',
        num_queries=900,
        as_two_stage=False,
        with_box_refine=False,
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        neck=dict(
            type='ChannelMapper',
            in_channels=[512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4
        ),
        encoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, batch_first=True),
                ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)
            )
        ),
        decoder=dict(
            num_layers=6,
            return_intermediate=True,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
                cross_attn_cfg=dict(embed_dims=256, batch_first=True),
                ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)
            ),
            post_norm_cfg=None
        ),
        positional_encoding=dict(type='SinePositionalEncoding',num_feats=128, normalize=True, offset=0.0, temperature=20),
        bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=num_classes,  # Replace this
        embed_dims=256,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0))
    ),
    # neck=None,  # Remove the original neck (now handled inside DeformableDETR)
    # topk_prompts=900,
    # num_feature_levels=5,
    # encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=5))),
    # decoder=dict(
    # layer_cfg=dict(
    #     cross_attn_cfg=dict(num_levels=5)
    # )),
    positional_encoding=dict(type='SinePositionalEncoding',num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(num_classes=num_classes)
)

# model = dict(
#     type='GroundingDINO',
#     backbone=dict(
#         type='DeformableDETR',
#         num_queries=300
#     ),
#     bbox_head=dict(num_classes=num_classes)
# )



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
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

max_epoch = 30

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            # frozen: top-level backbone of Grounding DINO (Swin-T)
            'backbone': dict(lr_mult=0.0),
            # Freeze the language model (CLIP)
            'language_model': dict(lr_mult=0.0),
             # Now we explicitly freeze def_detr parts except bbox_head
            'def_detr.backbone': dict(lr_mult=0.0),
            'def_detr.encoder': dict(lr_mult=0.0),
            'def_detr.decoder': dict(lr_mult=0.0),
            'def_detr.neck': dict(lr_mult=0.0)
        }))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
