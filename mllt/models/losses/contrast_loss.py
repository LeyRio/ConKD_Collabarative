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
import os


@LOSSES.register_module
class ContrastLoss(nn.Module):

    def __init__(self):
        super(ContrastLoss, self).__init__()
        

    # call: losses = self.contrastloss(query,keys,gt_labels,self.queue,self.queue_label) 
    def forward(self,
                query,
                keys,
                labels,
                queue,
                queue_label,
                K,
                T,
                BT,
                **kwargs):
        #loss = torch.square(cls_score - con_score)
        loss = 0
        # 如果一开始队列里面没有值该如何进行比较？
        #print(queue.shape)  # 128,512  512个128维向量
        
        q = queue.T
        l = torch.concat((labels,queue_label.T),0)

        # mask区分哪里是正例，哪里是负例
        # logits_mask构造分母求和控制矩阵(除了自身之外全部相加)
        torch.set_printoptions(threshold=np.inf)
        mask,weights = self.get_mask(labels,l)
        mask = torch.Tensor(mask).cuda()

        weights =torch.Tensor(weights).cuda()
        weights =torch.sqrt(weights/torch.max(weights))
        """
        with open('test.txt','a') as file0:
            print('labels',file=file0)
            print(labels,file=file0)
            print('l',file=file0)
            print(l,file=file0)
            print(mask,file=file0)
        """

        
        iddd = torch.rand(mask.shape[0],l.shape[0])
        logits_mask = torch.scatter(
            torch.ones_like(iddd),
            1,
            torch.arange(mask.shape[0]).view(-1, 1),
            0
        ).cuda()
        #print(logits_mask)
        # compute the similarity:
        # 首先组成用于参照的字典
        diction = torch.cat((keys,q),0)
        # 32 keys + 512 queue_keys
        # 计算整体相似度矩阵
        contrast_similarity = torch.div(torch.matmul(query, diction.T),T)
        logits_max, _ = torch.max(contrast_similarity, dim=1, keepdim=True)
        logits = contrast_similarity - logits_max.detach()

        #通过mask控制进行分母求和
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits  - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        log_prob_pos = (mask * log_prob)
        loss = - (T / BT) * log_prob_pos * weights

        return loss
    def get_mask(self,labels,q_labels):
        """
        function to get mask for loss computation
        We want to get the mask like this format:
        [0,1,0,0,1,1,0,0]
        [.....]
        """
        weights_mask = list()
        mask = list()
        for i in labels:
            c = list()
            w = list()
            #print(i)
            for j in q_labels:
                #print(j)
                weights = self.is_eq(i,j)
                w.append(weights)
                if weights > 0:
                    #print('True')
                    c.append(1)
                else:
                    #print('False')
                    c.append(0)
            mask.append(c)
            weights_mask.append(w)
        """
        a = torch.tensor([[1,0,1,0,1],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,1]])
        b = torch.tensor([[0,0,1,0,0],[1,0,1,1,1],[0,0,0,1,0]])
        q = list()
        for i in a:
            c = list()
            for j in b:
                if eq(i,j):
                    c.append(1)
                else:
                    c.append(0)
            q.append(c)
        print(q)
        [[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0]]
        """
        return mask,weights_mask
    def is_eq(self,label1,label2):
        label = (label1 + label2 == 2).sum()
        return label