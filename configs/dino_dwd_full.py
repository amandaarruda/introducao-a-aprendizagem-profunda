"""DINO detector campaign for full official DWD training splits."""
import os

default_scope = "mmdet"
custom_imports = dict(imports=["mmdet.datasets", "mmdet.models"], allow_failed_imports=False)
classes = ("bus", "bike", "car", "motor", "person", "rider", "truck")
dwd_root = os.environ["DWD_ROOT"]
train_domain = os.environ.get("TRAIN_DOMAIN", "day_clear")
max_iters = int(os.environ.get("MAX_ITERS", "30000"))
init_checkpoint = os.environ.get("INIT_CHECKPOINT", "")
r50_checkpoint = os.environ.get("DINO_R50_CHECKPOINT", "")
if not init_checkpoint and not r50_checkpoint:
    raise RuntimeError("DINO_R50_CHECKPOINT is required for the baseline")
load_from = init_checkpoint or None

model = dict(
    type="DINO", num_queries=300, with_box_refine=True, as_two_stage=True,
    data_preprocessor=dict(type="DetDataPreprocessor",
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True, pad_size_divisor=32),
    backbone=dict(type="ResNet", depth=50, num_stages=4, out_indices=(1, 2, 3),
                  frozen_stages=1, norm_cfg=dict(type="BN", requires_grad=False),
                  norm_eval=True, style="pytorch",
                  init_cfg=(None if init_checkpoint else
                            dict(type="Pretrained", checkpoint=r50_checkpoint))),
    neck=dict(type="ChannelMapper", in_channels=[512, 1024, 2048], kernel_size=1,
              out_channels=256, act_cfg=None, norm_cfg=dict(type="GN", num_groups=32),
              num_outs=4),
    encoder=dict(num_layers=6, layer_cfg=dict(
        self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
        ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(num_layers=6, return_intermediate=True, post_norm_cfg=None,
                 layer_cfg=dict(
                     self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                     cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                     ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(type="DINOHead", num_classes=len(classes), sync_cls_avg_factor=True,
                   loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                                 alpha=0.25, loss_weight=1.0),
                   loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                   loss_iou=dict(type="GIoULoss", loss_weight=2.0)),
    dn_cfg=dict(label_noise_scale=0.5, box_noise_scale=1.0,
                group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(assigner=dict(type="HungarianAssigner", match_costs=[
        dict(type="FocalLossCost", weight=2.0),
        dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
        dict(type="IoUCost", iou_mode="giou", weight=2.0),
    ])),
    test_cfg=dict(max_per_img=100, rcnn=dict(score_thr=0.05)),
)

train_pipeline = [dict(type="LoadImageFromFile"),
                  dict(type="LoadAnnotations", with_bbox=True),
                  dict(type="Resize", scale=(640, 640), keep_ratio=True),
                  dict(type="RandomFlip", prob=0.5), dict(type="PackDetInputs")]
test_pipeline = [dict(type="LoadImageFromFile"),
                 dict(type="Resize", scale=(640, 640), keep_ratio=True),
                 dict(type="LoadAnnotations", with_bbox=True),
                 dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape",
                                                        "img_shape", "scale_factor"))]
base_dataset = dict(type="CocoDataset", metainfo=dict(classes=classes), data_root="",
                     filter_cfg=dict(filter_empty_gt=False, min_size=1), serialize_data=False)
train_ann = os.path.join(dwd_root, train_domain, f"{train_domain}_train.coco.json")
val_ann = os.path.join(dwd_root, train_domain, f"{train_domain}_val.coco.json")
train_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True,
                        sampler=dict(type="DefaultSampler", shuffle=True),
                        dataset=dict(**base_dataset, ann_file=train_ann, data_prefix=dict(img=""),
                                     pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
                      sampler=dict(type="DefaultSampler", shuffle=False),
                      dataset=dict(**base_dataset, test_mode=True, ann_file=val_ann,
                                   data_prefix=dict(img=""), pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type="CocoMetric", ann_file=val_ann, metric="bbox",
                     metric_items=["mAP", "mAP_50"])
test_evaluator = val_evaluator
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=1e-4,
                      weight_decay=1e-4), clip_grad=dict(max_norm=0.1, norm_type=2),
                     paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}))
warmup_iters = max(1, min(1000, max_iters // 10))
param_scheduler = [dict(type="LinearLR", start_factor=0.01, by_epoch=False,
                        begin=0, end=warmup_iters),
                   dict(type="CosineAnnealingLR", T_max=max_iters-warmup_iters,
                        eta_min=1e-6, by_epoch=False, begin=warmup_iters, end=max_iters)]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=max_iters,
                                     by_epoch=False, save_last=True, max_keep_ckpts=1,
                                     save_optimizer=False),
                     logger=dict(type="LoggerHook", interval=20), timer=dict(type="IterTimerHook"))
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
               dist_cfg=dict(backend="nccl"))
randomness = dict(seed=1337, deterministic=False)
