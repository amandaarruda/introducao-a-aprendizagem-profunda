"""Additive DINO-R50 fine-tuning with batch 8; inherits the original detector unchanged."""

_base_ = ["./dino_dwd_finetune.py"]

# Keep the sample exposure aligned with the batch-1 campaign: validate per
# 1,000 training images and cap each candidate at 8,000 images.
max_iters = 1000
warmup_iters = 100
train_dataloader = dict(batch_size=8)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False,
         begin=0, end=warmup_iters),
    dict(type="CosineAnnealingLR", T_max=max_iters - warmup_iters,
         eta_min=1e-6, by_epoch=False, begin=warmup_iters, end=max_iters),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters,
                 val_interval=125)
default_hooks = dict(checkpoint=dict(
    interval=125,
    max_keep_ckpts=1,
    save_best="coco/bbox_mAP_50",
    rule="greater",
    save_optimizer=False,
))
custom_hooks = [dict(
    type="EarlyStoppingHook",
    monitor="coco/bbox_mAP_50",
    rule="greater",
    min_delta=0.001,
    patience=3,
    strict=True,
)]
