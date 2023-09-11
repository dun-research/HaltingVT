_base_ = ['../_base_videos_/default_runtime.py',
          'anet_roots.py',
          'data_pipeline_ml_10x64.py'
          ]

num_workers = 28
batch_size = 16
total_epoch = 50
warmup_epoch = 5
base_lr=1e-5
keep_rate = [1.0, 0.3]


# model settings
model = dict(
    type='HaltingVTRecognizer3D',
    backbone=dict(
        type='HaltingVT',
        pretrained_model="pretrained_weights/deit_small_patch16_224-cd65a155.pth",
        num_classes=0,   # Set cls_head here as Identity 
        num_frames=10,
        img_size=224,
        embed_dim=384,
        num_heads=6,
        divid_depth=len(keep_rate),
        keep_rate=keep_rate,
        get_idx=False,
        use_halting=True,
        use_distr_prior=False,
        halt_scale=10., 
        halt_center=10.,
        eps=0.01,
        use_motion_token=False,
        ),
    cls_head=dict(
        type='HaltingVTHead',
        num_classes=200,
        in_channels=384,
        average_clips='prob',
        use_motion_loss=True,
        use_motion_head=True, 
        loss_cls=dict(type='AdaHaltKLCELoss',
                        ponder_token_scale = 5e-4, 
                        motion_loss_beta=0.001,
                        )
        ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'RawframeDataset'
file_client_args = dict(io_backend='disk')

## data settings
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        ann_file=_base_.ann_file_train,
        data_prefix=dict(img=_base_.data_root),
        multi_class=True,
        num_classes=200,
        )
    )
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        ann_file=_base_.ann_file_val,
        data_prefix=dict(img=_base_.data_root_val),
        multi_class=True,
        num_classes=200,
        )
    )
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        ann_file=_base_.ann_file_test,
        data_prefix=dict(img=_base_.data_root_test),
        multi_class=True,
        num_classes=200,
        )
    )

val_evaluator = dict(type='AccMetric',
                        metric_list=(
                            'mean_average_precision'), 
                        )
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epoch, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
optim_wrapper = dict(
    optimizer=dict(
        type='Adam', lr=base_lr, weight_decay=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=warmup_epoch,
        convert_to_iter_based=False),
    dict(
        type='CosineAnnealingLR',
        T_max=total_epoch,
        eta_min=0.,
        by_epoch=True,
        begin=warmup_epoch,
        convert_to_iter_based=False)
]

_base_.default_hooks.update(
    logger=dict(type='LoggerHook', 
                interval=10,
                ignore_last=False,
                ),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto')
    )

# auto_scale_lr = dict(enable=False, base_batch_size=batch_size*8)
work_dir = 'work_dirs/HaltingVT_jointST_10x64x1_anet_deitSmall_G03_ML_Beta10'
