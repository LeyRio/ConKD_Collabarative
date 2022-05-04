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

class CSRAHead(nn.Module):
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRAHead, self).__init__()
        self.T = T      # temperature       
        self.lam = lam  # Lambda                        
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        if isinstance(x, tuple):
            x = x[-1]
        #print(x.shape)
        #x = x.view(x.size(0), -1,1,1)
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1)
        #print(score.shape)
        score = score.flatten(2)
        #print(score.shape)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99: # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit

@HEADS.register_module
class MulCSRAHead(nn.Module):
    """Simplest classification head, with only one fc layer for classification"""

    def __init__(self,
                in_channels=256,
                num_classes=80,
                method='fc',
                loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                loss_con = None,
                num_head = 1,
                lam = 0
                ):
        super(MulCSRAHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_channel = 256
        self.loss_cls = build_loss(loss_cls)
        self.loss_con = build_loss(loss_con)
        self.fc_cls = nn.Linear(self.fc_channel, num_classes)
        self.method = method
        self.no_bias = method in ['matmul', 'cos']
        self.block = CSRAHead
        self.tempreture_arch = {
            1: [1],
            2: [1, 99],
            4: [1, 2, 4, 99],
            6: [1, 2, 3, 4, 5, 99],
            8: [1, 2, 3, 4, 5, 6, 7, 99]
        }
        self.tem = self.tempreture_arch[num_head]
        self.multi_head = nn.ModuleList([
            CSRAHead(in_channels, num_classes, self.tem[i], lam)
            for i in range(num_head)
        ])


    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        if self.no_bias:
            self.fc_cls.bias.requires_grad=False
            print('No bias for classifier!')



    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
        

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
