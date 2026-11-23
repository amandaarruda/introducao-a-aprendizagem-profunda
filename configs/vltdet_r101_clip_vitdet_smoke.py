"""100-image ViTDet-style smoke test; not a benchmark/reproduction run."""
import os

custom_imports = dict(
    imports=["mmdet.datasets.transforms", "ligia_vltdet.models.openclip_resnet",
             "ligia_vltdet.models.vitdet_sfp"],
    allow_failed_imports=False,
)
default_scope = "mmdet"

classes = ("bus", "bike", "car", "motor", "person", "rider", "truck")
dwd_root = os.environ["DWD_ROOT"]
clip_checkpoint = os.environ.get("CLIP_RN101_CHECKPOINT", "")
train_domain = os.environ.get("TRAIN_DOMAIN", "day_clear")
max_iters = int(os.environ.get("MAX_ITERS", "500"))
warmup_iters = max(1, max_iters // 10)

model = dict(
    type="CascadeRCNN",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[122.771, 116.746, 104.094],
        std=[68.501, 66.632, 70.323],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="OpenCLIPModifiedResNet",
        checkpoint=clip_checkpoint,
        out_indices=(3,),
        norm_eval=True,
    ),
    neck=dict(
        type="ViTDetSimpleFeaturePyramid",
        in_channels=2048,
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
    ),
    roi_head=dict(
        type="CascadeRoIHead",
        num_stages=3,
        stage_loss_weights=[1.0, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64],
        ),
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=7,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=stds,
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            )
            for stds in ([0.1, 0.1, 0.2, 0.2],
                         [0.05, 0.05, 0.1, 0.1],
                         [0.033, 0.033, 0.067, 0.067])
        ],
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                          min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5,
                         neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0, pos_weight=-1, debug=False,
        ),
        rpn_proposal=dict(nms_pre=1000, max_per_img=1000,
                          nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=thr, neg_iou_thr=thr,
                              min_pos_iou=thr, match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1, debug=False,
            )
            for thr in (0.5, 0.6, 0.7)
        ],
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
    ),
)

load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
]
train_pipeline = [
    dict(type="Mosaic", img_scale=(640, 640), pad_val=114.0),
    dict(type="RandomAffine", border=(-320, -320), border_val=(114, 114, 114)),
    dict(type="MixUp", img_scale=(640, 640), pad_val=114.0),
    dict(type="RandAugment", aug_space=[
        [dict(type="Color")], [dict(type="Contrast")],
        [dict(type="Brightness")], [dict(type="Sharpness")],
    ], aug_num=2),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape",
                                           "img_shape", "scale_factor")),
]
base_dataset = dict(
    type="CocoDataset",
    metainfo=dict(classes=classes),
    data_root="",
    filter_cfg=dict(filter_empty_gt=False, min_size=1),
    serialize_data=False,
)
train_annotation = os.path.join(dwd_root, train_domain,
                                f"{train_domain}_train.coco.json")
val_annotation = os.path.join(dwd_root, train_domain,
                              f"{train_domain}_val.coco.json")
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="MultiImageMixDataset",
        dataset=dict(**base_dataset, ann_file=train_annotation,
                     data_prefix=dict(img=""), pipeline=load_pipeline),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(**base_dataset, test_mode=True,
                 ann_file=val_annotation,
                 data_prefix=dict(img=""), pipeline=test_pipeline),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="CocoMetric", ann_file=val_annotation,
                     metric="bbox", metric_items=["mAP_50"])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2),
    # The paper's layer-wise factor is interpreted stage-wise for this CNN
    # backbone: adjacent CLIP residual stages differ by 0.7. This is explicit
    # because the paper does not describe a ResNet-specific layer mapping.
    paramwise_cfg=dict(custom_keys={
        "backbone.visual.layer4": dict(lr_mult=1.0, decay_mult=1.0),
        "backbone.visual.layer3": dict(lr_mult=0.7, decay_mult=1.0),
        "backbone.visual.layer2": dict(lr_mult=0.7 ** 2, decay_mult=1.0),
        "backbone.visual.layer1": dict(lr_mult=0.7 ** 3, decay_mult=1.0),
        "backbone.visual": dict(lr_mult=0.7 ** 4, decay_mult=1.0),
    }),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=warmup_iters),
    dict(type="CosineAnnealingLR", T_max=max_iters-warmup_iters, eta_min=1e-6,
         by_epoch=False, begin=warmup_iters, end=max_iters),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=max_iters, by_epoch=False,
                    save_last=True, max_keep_ckpts=1, save_optimizer=False),
    logger=dict(type="LoggerHook", interval=10),
    timer=dict(type="IterTimerHook"),
)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
               dist_cfg=dict(backend="nccl"))
randomness = dict(seed=1337, deterministic=False)
