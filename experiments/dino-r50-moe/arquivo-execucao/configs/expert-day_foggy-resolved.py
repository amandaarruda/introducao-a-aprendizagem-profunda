base_dataset = dict(
    data_root='',
    filter_cfg=dict(filter_empty_gt=False, min_size=1),
    metainfo=dict(
        classes=(
            'bus',
            'bike',
            'car',
            'motor',
            'person',
            'rider',
            'truck',
        )),
    serialize_data=False,
    type='CocoDataset')
classes = (
    'bus',
    'bike',
    'car',
    'motor',
    'person',
    'rider',
    'truck',
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmdet.datasets',
        'mmdet.models',
    ])
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=12000,
        max_keep_ckpts=1,
        save_last=True,
        save_optimizer=False,
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
dwd_root = '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
init_checkpoint = '<PROJECT_ROOT>/outputs/dino-dayclear-full-16511/iter_30000.pth'
launcher = 'none'
load_from = '<PROJECT_ROOT>/outputs/dino-dayclear-full-16511/iter_30000.pth'
max_iters = 12000
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=None,
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=7,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=300,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=100, rcnn=dict(score_thr=0.05)),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.01, type='LinearLR'),
    dict(
        T_max=11000,
        begin=1000,
        by_epoch=False,
        end=12000,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
r50_checkpoint = '<PROJECT_ROOT>/checkpoints/resnet50-11ad3fa6.pth'
randomness = dict(deterministic=False, seed=1337)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_val.coco.json',
        data_prefix=dict(img=''),
        data_root='',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(
            classes=(
                'bus',
                'bike',
                'car',
                'motor',
                'person',
                'rider',
                'truck',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        serialize_data=False,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_val.coco.json',
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_ann = '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_train.coco.json'
train_cfg = dict(
    max_iters=12000, type='IterBasedTrainLoop', val_interval=12000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_train.coco.json',
        data_prefix=dict(img=''),
        data_root='',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(
            classes=(
                'bus',
                'bike',
                'car',
                'motor',
                'person',
                'rider',
                'truck',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        serialize_data=False,
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_domain = 'day_foggy'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_ann = '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_val.coco.json'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_val.coco.json',
        data_prefix=dict(img=''),
        data_root='',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(
            classes=(
                'bus',
                'bike',
                'car',
                'motor',
                'person',
                'rider',
                'truck',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        serialize_data=False,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_foggy/day_foggy_val.coco.json',
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
    ],
    type='CocoMetric')
warmup_iters = 1000
work_dir = '<PROJECT_ROOT>/outputs/dino-moe-experts-16511/day_foggy'
