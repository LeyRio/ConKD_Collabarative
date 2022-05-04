#coding=UTF-8
from __future__ import division

import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/home/liyou/DistributionBalancedLoss-master")
import argparse

import mmcv
from mmcv import Config, mkdir_or_exist
from mmcv.runner import load_checkpoint, get_dist_info

import os.path as osp
from mllt.datasets import build_dataset
from mllt.apis import (train_classifier, init_dist, get_root_logger,
                       set_random_seed)

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tools import utils

# maybe error
from mllt.apis.train import train_classifier_KD
import torch
from mllt.models import build_classifier
import shutil
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument(
        'config', help='train config file path')
    parser.add_argument(
        '--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    # todo: add validate
    parser.add_argument(
        '--validate', action='store_true', help='if validate when training')
    parser.add_argument(
        '--gpus', type=int, default=1, help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument(
        '--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    # step0:use parse
    args = parse_args()

    # step1:read config
    cfg = Config.fromfile(args.config)

    # ####### set some configurations

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # 通过设置cuda benchmark来为卷积网络搜索合适的方式从而加速运算
    # update configs according to CLI args
    # work_dir优先级： 命令行>配置文件
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 是否恢复上一次的训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # save config file to work dir
    mkdir_or_exist(cfg.work_dir)
    os.system('cp {} {}'.format(args.config, cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    # 如果 launcher 为 none，不启用分布式训练。不使用 dist_train.sh 默认参数为 none.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)

    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seedllL to {}'.format(args.seed))
        set_random_seed(args.seed)

    # step2:build dataset
    #train_dataset_pre = build_dataset(cfg.data_pre.train)
    train_dataset = build_dataset(cfg.data.train)

    # dump主要用于训练的时候拷贝数据
    if cfg.student_model.get('info_dir') is not None:
        mmcv.dump(dict(class_instance_num = train_dataset.class_instance_num.tolist()), osp.join(cfg.student_model.info_dir))

    # 构建预训练所需要模型
    #model_pre = build_classifier(cfg.model_pretrain, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # 构建教师模型
    teacher_model = build_classifier(cfg.pretrained_model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(teacher_model, cfg.pretrained_check_dir, map_location='cpu')
    # 构建学生模型
    student_model = build_classifier(cfg.student_model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    teacher_model.CLASSES = train_dataset.CLASSES
    student_model.CLASSES = train_dataset.CLASSES
    # step4:进行训练
    # 调用new函数训练复现的模型
    #print("******** begin pretrain *********")
    #train_classifier_pre(model_pre,train_dataset_pre,cfg,distributed=distributed, validate=args.validate, logger=logger)
    #logger.info(cfg.work_dir_pre)
    #checkpoint = load_checkpoint(model, cfg.pre_train_check_dir, map_location='cpu')
    train_classifier_KD(
        student_model,teacher_model, train_dataset, cfg,
        distributed=distributed, validate=args.validate, logger=logger)
    logger.info(cfg.work_dir)

if __name__ == '__main__':
    main()
