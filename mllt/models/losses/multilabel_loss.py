import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from .utils import weight_reduce_loss
from ..registry import LOSSES
from .cross_entropy_loss import cross_entropy, _expand_binary_labels, binary_cross_entropy, partial_cross_entropy
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn import Parameter
from sklearn.manifold import TSNE


@LOSSES.register_module
class MultiLabelLoss(nn.Module):

    def __init__(self,
                logit_reg=dict(
                                neg_scale=5.0,
                                init_bias=0.1
                            ),
                 freq_file='./class_freq.pkl'):
        super(MultiLabelLoss, self).__init__()

        self.class_freq = torch.from_numpy(np.asarray(
            mmcv.load(freq_file)['class_freq'])).float().cuda()
        self.neg_class_freq = torch.from_numpy(
            np.asarray(mmcv.load(freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        # 1.重新设置nk/n
        self.rou = self.class_freq/self.train_num

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq
        self.pos_sigmas = torch.nn.Parameter(torch.ones(self.init_bias.shape))
        self.pos_us = torch.nn.Parameter(torch.zeros(self.init_bias.shape))
        self.neg_sigmas = torch.nn.Parameter(torch.ones(self.init_bias.shape))
        self.neg_us = torch.nn.Parameter(torch.zeros(self.init_bias.shape))

    def forward(self,
                cls_score,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        crite = torch.nn.MultiLabelSoftMarginLoss()
        #weight = target * torch.exp(1-self.rou) + (1-target) * torch.exp(self.rou)
        #logits = target * (cls_score * self.pos_sigmas + self.pos_us) + (1-target) * (cls_score * self.neg_sigmas + self.neg_us)
        loss = crite(cls_score,target)
        #print(loss.shape)
        return loss