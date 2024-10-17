plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/sparse4dv3_temporal_r50_1x8_bs6_256x704'
total_batch_size = 48
num_gpus = 8
batch_size = 6
num_iters_per_epoch = 586
num_epochs = 100
checkpoint_epoch_interval = 20
checkpoint_config = dict(interval=11720)
log_config = dict(
    interval=51,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)
tracking_test = True
tracking_threshold = 0.2
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = 10
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = 4
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True
model = dict(
    type='Sparse4D',
    use_grid_mask=True,
    use_deformable_func=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        pretrained='ckpt/resnet50-19c8e357.pth'),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=0,
        out_channels=256,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    depth_branch=dict(
        type='DenseDepthNet',
        embed_dims=256,
        num_depth_layers=3,
        loss_weight=0.2),
    head=dict(
        type='Sparse4DHead',
        cls_threshold_to_reg=0.05,
        decouple_attn=True,
        instance_bank=dict(
            type='InstanceBank',
            num_anchor=900,
            embed_dims=256,
            anchor='nuscenes_kmeans900.npy',
            anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
            num_temp_instances=600,
            confidence_decay=0.6,
            feat_grad=False),
        anchor_encoder=dict(
            type='SparseBox3DEncoder',
            vel_dims=3,
            embed_dims=[128, 32, 32, 64],
            mode='cat',
            output_fc=False,
            in_loops=1,
            out_loops=4),
        num_single_frame_decoder=1,
        operation_order=[
            'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
            'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
            'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
            'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
            'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
            'deformable', 'ffn', 'norm', 'refine'
        ],
        temp_graph_model=dict(
            type='MultiheadAttention',
            embed_dims=512,
            num_heads=8,
            batch_first=True,
            dropout=0.1),
        graph_model=dict(
            type='MultiheadAttention',
            embed_dims=512,
            num_heads=8,
            batch_first=True,
            dropout=0.1),
        norm_layer=dict(type='LN', normalized_shape=256),
        ffn=dict(
            type='AsymmetricFFN',
            in_channels=512,
            pre_norm=dict(type='LN'),
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True)),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode='cat',
            kps_generator=dict(
                type='SparseBox3DKeyPointsGenerator',
                num_learnable_pts=6,
                fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                           [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                           [0, 0, -0.45]])),
        refine_layer=dict(
            type='SparseBox3DRefinementModule',
            embed_dims=256,
            num_cls=10,
            refine_yaw=True,
            with_quality_estimation=True),
        sampler=dict(
            type='SparseBox3DTarget',
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            cls_wise_reg_weights=dict(
                {9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_reg=dict(
            type='SparseBox3DLoss',
            loss_box=dict(type='L1Loss', loss_weight=0.25),
            loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True),
            loss_yawness=dict(type='GaussianFocalLoss'),
            cls_allow_reverse=[5]),
        decoder=dict(type='SparseBox3DDecoder'),
        reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
dataset_type = 'NuScenes3DDetTrackDataset'
data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_anno_pkls/'
file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='ResizeCropFlipImage'),
    dict(type='MultiScaleDepthMapGenerator', downsample=[4, 8, 16]),
    dict(type='BBoxRotation'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth',
            'focal', 'gt_bboxes_3d', 'gt_labels_3d'
        ],
        meta_keys=['T_global', 'T_global_inv', 'timestamp', 'instance_id'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
        meta_keys=['T_global', 'T_global_inv', 'timestamp'])
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
data_basic_config = dict(
    type='NuScenes3DDetTrackDataset',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval')
data_aug_conf = dict(
    resize_lim=(0.4, 0.47),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[-0.3925, 0.3925])
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        type='NuScenes3DDetTrackDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/nuscenes_anno_pkls/nuscenes_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='ResizeCropFlipImage'),
            dict(type='MultiScaleDepthMapGenerator', downsample=[4, 8, 16]),
            dict(type='BBoxRotation'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='CircleObjectRangeFilter',
                class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
            dict(
                type='InstanceNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'gt_depth', 'focal', 'gt_bboxes_3d', 'gt_labels_3d'
                ],
                meta_keys=[
                    'T_global', 'T_global_inv', 'timestamp', 'instance_id'
                ])
        ],
        test_mode=False,
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[-0.3925, 0.3925]),
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True),
    val=dict(
        type='NuScenes3DDetTrackDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/nuscenes_anno_pkls/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
                meta_keys=['T_global', 'T_global_inv', 'timestamp'])
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[-0.3925, 0.3925]),
        test_mode=True,
        tracking=True,
        tracking_threshold=0.2),
    test=dict(
        type='NuScenes3DDetTrackDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/nuscenes_anno_pkls/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
                meta_keys=['T_global', 'T_global_inv', 'timestamp'])
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[-0.3925, 0.3925]),
        test_mode=True,
        tracking=True,
        tracking_threshold=0.2))
optimizer = dict(
    type='AdamW',
    lr=0.0006,
    weight_decay=0.001,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.5))))
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='IterBasedRunner', max_iters=58600)
vis_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='Collect', keys=['img'], meta_keys=['timestamp', 'lidar2img'])
]
evaluation = dict(
    interval=11720,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='Collect', keys=['img'], meta_keys=['timestamp', 'lidar2img'])
    ])
gpu_ids = range(0, 1)
