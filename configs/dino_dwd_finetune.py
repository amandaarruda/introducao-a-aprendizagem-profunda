"""Additive DINO-R50 fine-tuning config; inherits the original detector unchanged."""
import os

_base_ = ["./dino_dwd_full.py"]

max_iters = 8000
warmup_iters = 800
finetune_lr = float(os.environ["FINETUNE_LR"])
init_checkpoint = os.environ["INIT_CHECKPOINT"]
load_from = init_checkpoint

# Keep the inherited model, data, augmentations, loss, batch size, and seed.
optim_wrapper = dict(optimizer=dict(lr=finetune_lr))
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False,
         begin=0, end=warmup_iters),
    dict(type="CosineAnnealingLR", T_max=max_iters - warmup_iters,
         eta_min=1e-6, by_epoch=False, begin=warmup_iters, end=max_iters),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=1000)
default_hooks = dict(checkpoint=dict(
    interval=1000,
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
