"""Longer 640px DayClear baseline with reserved validation each 1k updates."""
_base_ = "./vltdet_r101_clip_vitdet_smoke.py"

# MAX_ITERS=6000 is set by the launcher; evaluate on reserved model-validation
# at fixed intervals and preserve the best validation checkpoint.
train_cfg = dict(val_interval=1000)
default_hooks = dict(checkpoint=dict(
    interval=1000,
    max_keep_ckpts=2,
    save_best="coco/bbox_mAP_50",
    rule="greater",
))
