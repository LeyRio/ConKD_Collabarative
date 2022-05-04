# model settings

pretrained_model = dict(
    type='GCNSimpleClassifier',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(0, 1, 2),
        frozen_stages=3,
        style='pytorch'),
    neck1=dict(
        type='ResNetLast',
        depth=50,
        num_stages=1,
        strides=(2,),
        dilations=(1,),
        out_indices=(0,),
        frozen_stages=1,
        style='pytorch'),
    neck2 = dict(
        type='ResNetLast',
        depth=50,
        num_stages=1,
        strides=(2,),
        dilations=(1,),
        out_indices=(0,),
        frozen_stages=1,
        style='pytorch'),
    head=dict(
        type='DualHead',
        in_channels=2048,
        num_classes=80,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=5),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0, freq_file='appendix/coco/longtail2017/class_freq.pkl'),
        loss_con = dict(type='ConCollaLoss')),
        #CollaborateLoss    
        #ResampleLoss
    head2=dict(
        type='DualHeadRandom',
        in_channels=2048,
        num_classes=80,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=5),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0, freq_file='appendix/coco/longtail2017/class_freq.pkl'),
        loss_con = dict(type='ConCollaLoss')),
    gcn1=dict(
        type='GcnHead',
        in_channels=2048,
        num_classes=80,
        word2vec_len=300,
        mid_channels=1024,
        t=0,
        ave_pool=False,
        adj_file='appendix/coco/coco_adj.pkl',
        word2vec_file='appendix/coco/coco_glove_word2vec.pkl',
        loss_cls = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    gcn2=dict(
        type='GcnHead',
        in_channels=2048,
        num_classes=80,
        word2vec_len=300,
        mid_channels=1024,
        t=0,
        ave_pool=False,
        adj_file='appendix/coco/coco_adj.pkl',
        word2vec_file='appendix/coco/coco_glove_word2vec.pkl',
        loss_cls = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    multilable_loss = dict(type='MultiLabelLoss',logit_reg=dict(init_bias=0.05, neg_scale=5),freq_file='appendix/coco/longtail2017/class_freq.pkl')    
        )
student_model = dict(
    type='DualKDClassifier',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(0, 1, 2),
        frozen_stages=3,
        style='pytorch'),
    neck1=dict(
        type='ResNetLast',
        depth=50,
        num_stages=1,
        strides=(2,),
        dilations=(1,),
        out_indices=(0,),
        frozen_stages=1,
        style='pytorch'),
    neck2 = dict(
        type='ResNetLast',
        depth=50,
        num_stages=1,
        strides=(2,),
        dilations=(1,),
        out_indices=(0,),
        frozen_stages=1,
        style='pytorch'),
    head=dict(
        type='DualHead',
        in_channels=2048,
        num_classes=80,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=5),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0, freq_file='appendix/coco/longtail2017/class_freq.pkl'),
        loss_con = dict(type='ConCollaLoss')),
        #CollaborateLoss    
        #ResampleLoss
    head2=dict(
        type='DualHeadRandom',
        in_channels=2048,
        num_classes=80,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=5),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0, freq_file='appendix/coco/longtail2017/class_freq.pkl'),
        loss_con = dict(type='ConCollaLoss')),
        T = 15,
        lamda = 10,
        logit_reg=dict(init_bias=0.05, neg_scale=5),
        freq_file='appendix/coco/longtail2017/class_freq.pkl')
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mnt/SSD/det/coco/'
online_data_root = 'appendix/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)

img_size=224
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler='ClassAware',
    train=dict(
            type=dataset_type,
            ann_file=online_data_root + 'annotations/instances_train2017.json',
            LT_ann_file = [online_data_root + 'longtail2017/img_id.pkl'],
            img_prefix=online_data_root + 'train2017/',
            img_scale=(img_size, img_size),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5),
    val=dict(
        type=dataset_type,
        ann_file=online_data_root + 'annotations/instances_val2017.json',
        img_prefix=online_data_root + 'val2017/',
        img_scale=(img_size, img_size),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
        type=dataset_type,
        ann_file=online_data_root + 'annotations/instances_val2017.json',
        class_split=online_data_root + 'longtail2017/class_split.pkl',
        img_prefix=online_data_root + 'val2017/',
        img_scale=(img_size, img_size),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32, 
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
optimizer = dict(type='SGD', lr=0.33, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#pretrained_check_dir = 'work_dirs/LT_coco_resnet50_gcn/epoch_8.pth'
pretrained_check_dir = 'work_dirs/LT_coco_resnet50_contrast_coco/epoch_8.pth'
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=700,
    warmup_ratio=1.0 / 3,
    step=[5,7])  # 8: [5,7]) 4: [2,3]) 40: [25,35]) 80: [55,75])
checkpoint_config = dict(interval=8)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
start_epoch=0
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/LT_coco_resnet50_KD'
load_from = None
if start_epoch > 0:
    resume_from = work_dir + '/epoch_{}.pth'.format(start_epoch)
    print("start from epoch {}".format(start_epoch))
else:
    resume_from = None
workflow = [('train', 1)]