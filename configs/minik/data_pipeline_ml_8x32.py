
# dataset settings
dataset_type = 'RawFrameFakeVidDataset'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='RandAug', aug_type="base"),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=['img_shape', 'video_id', 'motion_label', 'frame_dir'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=['img_shape', 'video_id', 'motion_label', 'frame_dir'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', meta_keys=['img_shape', 'video_id', 'motion_label', 'frame_dir'])
]
train_dataloader = dict(
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        filename_tmpl="img_{:05}.jpg",
        start_index=1,
        pipeline=train_pipeline,
        from_same_vid=True,
        ))
val_dataloader = dict(
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        filename_tmpl="img_{:05}.jpg",
        start_index=1,
        pipeline=val_pipeline,
        from_same_vid=True,
        test_mode=True))
test_dataloader = dict(
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        filename_tmpl="img_{:05}.jpg",
        start_index=1,
        pipeline=test_pipeline,
        from_same_vid=True,
        test_mode=True))