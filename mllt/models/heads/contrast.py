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
class ContrastHead(nn.Module):
    """Simplest classification head, with only one fc layer for classification"""

    def __init__(self,
                in_channels=256,
                num_classes=80,
                method='fc'
                ):
        super(ContrastHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_channel = in_channels
        self.fc_cls = nn.Linear(self.fc_channel, num_classes)
        self.method = method
        self.no_bias = method in ['matmul', 'cos']

        ## 
        #construct the mlp structure
        self.fc1 = nn.Linear(in_channels, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.act = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(1024,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU(inplace = True)



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

        #init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        #init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        # return the feature extracted
        if isinstance(x, tuple):
            x = x[-1]
        #print(x.shape)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        #x = self.act2(x)
        #x = self.drop(x)
        
        return x


    def nonzero_cosine_similarity(self, x1, x2):
        # mask = x1 != 0
        cls_score = torch.zeros(x1.shape[0], self.num_classes).cuda()
        for i, x in enumerate(x1):
            mask = (x != 0).float()
            # print(torch.sum(mask))
            cls_score[i] = torch.cosine_similarity(x.unsqueeze(0),x2*mask)

        return cls_score
