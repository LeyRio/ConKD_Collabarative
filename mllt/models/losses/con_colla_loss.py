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


# 这个损失求解比较简单
@LOSSES.register_module
class ConCollaLoss(nn.Module):

    def __init__(self):
        super(ConCollaLoss, self).__init__()

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))

    def forward(self,
                cls_score,
                con_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        loss = torch.square(cls_score - con_score)
        #print(loss.shape)
        return loss