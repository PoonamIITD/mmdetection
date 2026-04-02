_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']






#remove later!!!!!!!!!!!!!!
resume = True







# Pretrained weights
# load_from = '/home/tushar/scratch/checkpoints/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
load_from = '/home/poonam_rajput/scratch/mmdetection/checkpoints/co-dino_rsud_finetuned_best_coco_bbox_mAP_epoch_18.pth'
data_root = '/home/poonam_rajput/scratch/dataset/RSUD_dataset/'

# Custom classes
classes = (
    "person", "rickshaw", "rickshaw van", "auto rickshaw", "truck", "pickup truck",
    "private car", "motorcycle", "bicycle", "bus", "micro bus", "covered van", "human hauler"
)

# in your config file (e.g. co_dino_custom.py)
# custom_imports = dict(
#     imports=['projects.CODETR.codetr.co_dump_head'],
#     allow_failed_imports=False
# )



# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = len(classes)
# num_classes = 80

# ====================
# Model settings
# ====================
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        _delete_=True,
        type='CoDINOHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ]
)

# ====================
# Pipelines
# ====================
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
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048), (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048), (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ====================
# Dataloaders
# ====================
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes))
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/merged_instances_val2017_codetr_final.json',
        data_prefix=dict(img='images/val'),
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes))
)


test_dataloader = val_dataloader


# ====================
# Evaluators
# ====================

#tentative
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/merged_instances_val2017_codetr_final.json',
    metric='bbox')


val_cfg = dict(type='ValLoop')

test_evaluator = val_evaluator 


test_cfg = dict(type='TestLoop')

# ====================
# Training schedule
# ====================
max_epochs = 49

optim_wrapper = dict(optimizer=dict(lr=1e-4))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, save_best='auto', max_keep_ckpts=3)
)

log_processor = dict(by_epoch=True)
