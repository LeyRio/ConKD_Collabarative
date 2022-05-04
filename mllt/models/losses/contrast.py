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
class ContrastLossV2(nn.Module):

    def __init__(self,
            temperature=0.07, 
            contrast_mode='all',
            base_temperature=0.07, 
            match_type='all'):
        super(ContrastLossV2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.match_type = match_type

    # call: losses = self.contrastloss(query,keys,gt_labels,self.queue,self.queue_label) 
    def forward(self, features, labels=None, mask=None):
        torch.set_printoptions(threshold=np.inf)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            #labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #mask = torch.eq(labels, labels.T).float().to(device)
            batch_size = labels.shape[0]
            mask = torch.zeros((batch_size, batch_size)).to(device)

            # populate mask for complete match 
            labels = labels.cpu().numpy()
            for i in range(len(labels)):
                for j in range(len(labels)):

                    gt_i = labels[i]
                    gt_j = labels[j]

                    # different ways to handle multiclass-multilabel 
                    if self.match_type == "all":
                        mask[i,j] = 1 if (gt_i == gt_j).all() else 0 
                    elif self.match_type == "any":
                        pos_idx = (gt_i == 1) | (gt_j == 1)
                        gt_i = gt_i[pos_idx]
                        gt_j = gt_j[pos_idx]
                        mask[i,j] = 1 if (gt_i == gt_j).any() else 0 
                    elif self.match_type == "iou_weighted":
                        pos_idx = (gt_i == 1) | (gt_j == 1)
                        gt_i = gt_i[pos_idx]
                        gt_j = gt_j[pos_idx]
                        if pos_idx.sum() == 0:
                            weight = 0
                        else: 
                            weight = (gt_i == gt_j).sum() / pos_idx.sum()
                        mask[i,j] = weight
                    elif self.match_type == "f1_weighted":
                        pos_idx = (gt_i == 1) | (gt_j == 1)
                        gt_i = gt_i[pos_idx]
                        gt_j = gt_j[pos_idx]
                        tp = (gt_i == gt_j).sum()
                        fp = gt_i.sum() - tp
                        fn = gt_j.sum() - tp
                        weight = (2*tp) / (2*tp + fp + fn)
                        mask[i,j] = weight
                    elif self.match_type == "one_weighted":
                        pos_idx = (gt_i == 1) | (gt_j == 1)
                        gt_i = gt_i[pos_idx]
                        gt_j = gt_j[pos_idx]
                        weight = (gt_i == gt_j).sum() / len(gt_i)
                        mask[i,j] = weight
                    elif self.match_type == "zero_and_one_weighted":
                        mask[i,j] = (gt_i == gt_j).sum() / len(gt_i)
                    else:
                        print(f"match type {self.match_type} is not implemented")
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask).to(device),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        # 可能是logits_mask存在问题导致出现nan
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = - (self.temperature / self.base_temperature) * (mask * log_prob)
        mean_log_prob_posw = (mask * log_prob).sum(1) / (mask.sum(1))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_posw
        loss = loss.view(anchor_count, batch_size).mean()
        """
        with open('test.txt','a') as file0:
            print(logits,file = file0)
            print(exp_logits,file = file0)
            print(log_prob,file=file0)
            print(logits_mask,file = file0)
            print("loss:",file=file0)
            print(loss,file=file0)
        """
        

        return mean_log_prob_pos