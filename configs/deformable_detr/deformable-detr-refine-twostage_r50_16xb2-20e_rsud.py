_base_ = 'deformable-detr-refine_r50_16xb2-50e_coco.py'
model = dict(as_two_stage=True)

load_from = "/home/poonam_rajput/scratch/mmdetection/checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth"
max_epochs = 20

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

dataset_type = 'CocoDataset'
data_root = '/home/poonam_rajput/scratch/dataset/RSUD_dataset'   # path to dataset root
images_data_root = '/home/poonam_rajput/scratch/dataset/RSUD_dataset/images'

# class names
classes = ('person', 'rickshaw', 'rickshaw van', 'auto rickshaw',
           'truck', 'pickup truck', 'private car', 'motorcycle',
           'bicycle', 'bus', 'micro bus', 'covered van', 'human hauler')

metainfo = dict(classes=classes)

model = dict(
    bbox_head=dict(num_classes=len(classes))
)


train_ann_file = data_root + '/annotations/instances_train2017.json'
val_ann_file = data_root + '/annotations/instances_val2017.json'

train_img_prefix = images_data_root + '/train/'   # path where train images are stored
val_img_prefix = images_data_root + '/val/'       # path where val images are stored

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_prefix),
        test_mode=True
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',  # or your metric
        rule='greater'              # 'less' for losses
    )
)