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
clip_checkpoint = '<PROJECT_ROOT>/checkpoints/RN101.pt'
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.datasets.transforms',
        'ligia_vltdet.models.openclip_resnet',
        'ligia_vltdet.models.vitdet_sfp',
    ])
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/bbox_mAP_50',
        save_last=True,
        save_optimizer=False,
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
dwd_root = '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
max_iters = 30000
model = dict(
    backbone=dict(
        checkpoint='<PROJECT_ROOT>/checkpoints/RN101.pt',
        norm_eval=True,
        out_indices=(3, ),
        type='OpenCLIPModifiedResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            122.771,
            116.746,
            104.094,
        ],
        pad_size_divisor=32,
        std=[
            68.501,
            66.632,
            70.323,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=2048,
        out_channels=256,
        scale_factors=(
            4.0,
            2.0,
            1.0,
            0.5,
        ),
        type='ViTDetSimpleFeaturePyramid'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=7,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=7,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=7,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                8,
                16,
                32,
                64,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1.0,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                4,
            ],
            strides=[
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    type='CascadeRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.visual':
            dict(decay_mult=1.0, lr_mult=0.24009999999999995),
            'backbone.visual.layer1':
            dict(decay_mult=1.0, lr_mult=0.3429999999999999),
            'backbone.visual.layer2':
            dict(decay_mult=1.0, lr_mult=0.48999999999999994),
            'backbone.visual.layer3':
            dict(decay_mult=1.0, lr_mult=0.7),
            'backbone.visual.layer4':
            dict(decay_mult=1.0, lr_mult=1.0)
        })),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=3000, start_factor=0.01, type='LinearLR'),
    dict(
        T_max=27000,
        begin=3000,
        by_epoch=False,
        end=30000,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=1337)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_val.coco.json',
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
    '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_val.coco.json',
    metric='bbox',
    metric_items=[
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
train_annotation = '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_train.coco.json'
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            ann_file=
            '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_train.coco.json',
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
            ],
            serialize_data=False,
            type='CocoDataset'),
        pipeline=[
            dict(img_scale=(
                640,
                640,
            ), pad_val=114.0, type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                type='RandomAffine'),
            dict(img_scale=(
                640,
                640,
            ), pad_val=114.0, type='MixUp'),
            dict(
                aug_num=2,
                aug_space=[
                    [
                        dict(type='Color'),
                    ],
                    [
                        dict(type='Contrast'),
                    ],
                    [
                        dict(type='Brightness'),
                    ],
                    [
                        dict(type='Sharpness'),
                    ],
                ],
                type='RandAugment'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='MultiImageMixDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_domain = 'day_clear'
train_pipeline = [
    dict(img_scale=(
        640,
        640,
    ), pad_val=114.0, type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        type='RandomAffine'),
    dict(img_scale=(
        640,
        640,
    ), pad_val=114.0, type='MixUp'),
    dict(
        aug_num=2,
        aug_space=[
            [
                dict(type='Color'),
            ],
            [
                dict(type='Contrast'),
            ],
            [
                dict(type='Brightness'),
            ],
            [
                dict(type='Sharpness'),
            ],
        ],
        type='RandAugment'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_annotation = '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_val.coco.json'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_val.coco.json',
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
    '<PROJECT_ROOT>/data/raw/dwd/full-train-frozen-eval-20261005/day_clear/day_clear_val.coco.json',
    metric='bbox',
    metric_items=[
        'mAP_50',
    ],
    type='CocoMetric')
warmup_iters = 3000
work_dir = '<PROJECT_ROOT>/outputs/vitdet-r101-dayclear-full-640-30000-32364'
