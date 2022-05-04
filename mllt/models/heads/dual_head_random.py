import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from torch.nn import Parameter
import numpy as np
import mmcv
from torch.nn import init

@HEADS.register_module
class DualHeadRandom(nn.Module):
    """Simplest classification head, with only one fc layer for classification"""

    def __init__(self,
                in_channels=256,
                num_classes=80,
                method='fc',
                loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                loss_con = None
                ):
        super(DualHeadRandom, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_channel = 256
        self.loss_cls = build_loss(loss_cls)
        self.loss_con = build_loss(loss_con)
        self.fc_cls = nn.Linear(self.fc_channel, num_classes)
        self.method = method
        self.no_bias = method in ['matmul', 'cos']

        ## 
        
        self.fc1 = nn.Linear(in_channels, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)



    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        if self.no_bias:
            self.fc_cls.bias.requires_grad=False
            print('No bias for classifier!')
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        #init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        #init.constant_(self.fc1.bias, 0)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        #print(x.shape)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn(x)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        #x = self.drop(x)
        
        
        if self.method in ['fc','matmul']:
            cls_score = self.fc_cls(x)
        elif self.method == 'cos':
            cls_score = torch.cosine_similarity(x.unsqueeze(-1), self.fc_cls.weight.t().unsqueeze(0), dim=1)
        elif self.method == 'no_zero':
            cls_score = self.nonzero_cosine_similarity(x, self.fc_cls.weight)
        elif self.method == 'norm_w':
            cls_weight = self.fc_cls.weight / torch.norm(self.fc_cls.weight, 2, 1, keepdim=True)
            cls_score = torch.mm(x, cls_weight.t()) + self.fc_cls.bias
        else:
            raise NameError
        return cls_score

    def loss(self,
             cls_score,
             con_score,
             labels,
             weight=None,
             reduction_override=None):
        losses = dict()
        ## TODO
        
        losses['loss_cls'] = self.loss_cls(
        cls_score,
        labels,
        weight,
        avg_factor=None,
        reduction_override=reduction_override) 
        losses['loss_con'] = 0.1 * self.loss_con(cls_score,con_score,labels,weight,avg_factor=None,reduction_override=reduction_override)
        
        """
        
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            weight,
            avg_factor=None,
            reduction_override=reduction_override)
        """
        if isinstance(cls_score, list):
            cls_score = cls_score[0]
        losses['acc'] = accuracy(cls_score, labels)
        return losses

    def nonzero_cosine_similarity(self, x1, x2):
        # mask = x1 != 0
        cls_score = torch.zeros(x1.shape[0], self.num_classes).cuda()
        for i, x in enumerate(x1):
            mask = (x != 0).float()
            # print(torch.sum(mask))
            cls_score[i] = torch.cosine_similarity(x.unsqueeze(0),x2*mask)

        return cls_score
