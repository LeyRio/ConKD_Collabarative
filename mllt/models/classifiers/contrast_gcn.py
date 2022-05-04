import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
import copy
import torch.nn.functional as F

@CLASSIFIERS.register_module
class ContrastGCNClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck1,
                 neck2,
                 head,
                 head2,
                 mlp1,
                 mlp2,
                 gcn1,
                 gcn2,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 savefeat=False,
                 contrast_loss = None,
                 multilable_loss = None,
                 classes = 20):
        super(ContrastGCNClassifier, self).__init__()
        # just resnet50 except last stage
        self.backbone = builder.build_backbone(backbone)
        # neck for main network:
        self.neck1 = builder.build_neck(neck1)
        self.neck2 = builder.build_neck(neck2)

        # mlp for contrastive:
        self.mlp_q = builder.build_head(mlp1)
        self.gcn1 = builder.build_head(gcn1)
        self.gcn2 = builder.build_head(gcn2)

        #head for contrastive:
        self.head1 = builder.build_head(head)
        self.head2 = builder.build_head(head2)
        self.contrastloss = builder.build_loss(contrast_loss)
        self.multilabel_loss = builder.build_loss(multilable_loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.lock_back=lock_back
        self.lock_neck=lock_neck
        self.savefeat=savefeat
        self.lamda = 1.0

        if self.savefeat and not self.with_neck:
            assert neck1 is not None, 'We must have a neck'
            assert train_cfg is None, 'this is only at testing stage'
        if self.lock_back:
            print('\033[1;35m >>> backbone locked !\033[0;0m')
        if self.lock_neck:
            print('\033[1;35m >>> neck locked !\033[0;0m')
        self.init_weights(pretrained=pretrained)
        self.count = CountMeter(num_classes=20)

        # freq = torch.tensor([5,6,6,5,7,16,12,5,9,6,5,5,10,21,6,5],dtype=torch.float)
        # self.cla_weight = torch.mean(torch.sqrt(freq))*(torch.ones(freq.shape,dtype=torch.float) / torch.sqrt(freq)).cuda()
        # print(self.cla_weight)


    def init_weights(self, pretrained=None):
        super(ContrastGCNClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.mlp_q.init_weights()
        self.neck1.init_weights(pretrained=pretrained)
        self.neck2.init_weights(pretrained=pretrained)
        self.head1.init_weights()
        self.head2.init_weights()
        self.gcn1.init_weights()
        self.gcn2.init_weights()


    # return self.forward_train(img, img_meta, **kwargs)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      flag,
                      epoches = None):
        if flag == 1:
            # data from CA loader,Need to split into two imgs:
            targets = copy.deepcopy(gt_labels)
            targets = torch.stack(targets)
            targets = targets.transpose_(0,1)

            # stack inputs
            inputs = torch.cat([img[0], img[1]], dim=0)
            inputs = inputs.cuda()
            #print(targets.shape)
            #print(type(gt_labels))
            # first under backbone:
            x1 = self.backbone(inputs)
            # For contrastive:
            x = self.neck1(x1)
            feat = self.mlp_q(x)
            #query = nn.functional.normalize(feat1, dim=1)
            batch_size = targets.shape[0]
            f1, f2 = torch.split(feat, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            features = nn.functional.normalize(features, dim=2)
            
            # 是否可以考虑和另一个分支的输出结果进行对比？
            #For main module:
            img1 = img[0]
            img2 = img[1]
            gt_labels = gt_labels[0]
            x_1 = self.backbone(img1)
            x_copy = x_1
            # 第一分支输出
            outs = self.neck1(x_1)
            outs1 = self.head1(outs)
            if isinstance(outs, tuple):
                outs = outs[-1]
            #print(x.shape)
            outs = F.avg_pool2d(outs, outs.size()[2:])
            outs = outs.view(outs.size(0), -1)
            outs2 = self.gcn1(outs)
            # 第二分支输出
            outs_copy = self.neck2(x_copy)
            outs3 = self.head2(outs_copy)
            if isinstance(outs_copy, tuple):
                outs_copy = outs_copy[-1]
            #print(x.shape)
            outs_copy = F.avg_pool2d(outs_copy, outs_copy.size()[2:])
            outs_copy = outs_copy.view(outs_copy.size(0), -1)
            outs4 = self.gcn2(outs_copy)
            # 得到loss,传入之后会计算cls_loss + con_loss
            loss_inputs1 = (outs1, outs3, gt_labels)
            losses = self.head1.loss(*loss_inputs1)
            #print(self.contrastloss(features,targets) )
            # Add contrastive loss:
            losses['loss_contrastive'] =self.contrastloss(features,targets) 
            losses['loss_gcn'] = 0.1 * (self.multilabel_loss(outs2,gt_labels) + self.multilabel_loss(outs4,gt_labels))
            
        else:
            #data from Random loader,just one picture
            y = self.backbone(img)
            y_copy = y
            # 第二分支输出
            #print("data2 for head2")
            outs_y = self.neck2(y)
            outs_y1 = self.head2(outs_y)
            if isinstance(outs_y, tuple):
                outs_y = outs_y[-1]
            #print(x.shape)
            outs_y = F.avg_pool2d(outs_y, outs_y.size()[2:])
            outs_y = outs_y.view(outs_y.size(0), -1)
            outs_y2 = self.gcn2(outs_y)
            # 第一分支输出
            #print("data2 for head1")
            outs_copy_y = self.neck1(y_copy)
            outs2_y1 = self.head1(outs_copy_y)
            if isinstance(outs_copy_y, tuple):
                outs_copy_y = outs_copy_y[-1]
            #print(x.shape)
            outs_copy_y = F.avg_pool2d(outs_copy_y, outs_copy_y.size()[2:])
            outs_copy_y = outs_copy_y.view(outs_copy_y.size(0), -1)
            outs2_y2 = self.gcn1(outs_copy_y)

            loss_inputs = (outs_y1,outs2_y1,gt_labels)
            losses = self.head2.loss(*loss_inputs)
            losses['loss_gcn'] = 0.1 * (self.multilabel_loss(outs_y2,gt_labels) + self.multilabel_loss(outs2_y2,gt_labels))
        return losses

    # def simple_test(self, img, img_meta=None, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.head(x)
    #     return outs

# TODO If Need test?
    def simple_test(self, img, img_meta, rescale=False):
        x = self.backbone(img)
        y = x
        outs = self.neck1(x)
        outs = self.head1(outs)
        outs2 = self.neck2(y)
        outs2 = self.head2(outs2)
        
        
        #outs3 = (outs + outs2)/2
        outs3 = (outs2 + outs)/2
        if self.savefeat:
            return outs3, x

        return outs3

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

''' This part is only for resampling
'''
class CountMeter(object):
    def __init__(self, num_classes, vector_len=1, end_n=15500): # coco reduce 4: 22560 # voc reduce 1: 15500
        self.num_classes = num_classes
        self.end_n = end_n
        self.vector_len = vector_len
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.num_classes, dtype=torch.int64).cuda()
        self.all_features = None
        self.all_labels = None
        self.n = 0

    def update(self, gt_labels, features=None):
        n = gt_labels.shape[0]
        self.count += torch.sum(gt_labels, dim=0)
        if self.all_labels is None:
            self.all_labels = gt_labels.cpu().numpy()
            self.all_features = features.cpu().numpy()
        else:
            self.all_labels = np.concatenate((self.all_labels, gt_labels.cpu().numpy()))
            self.all_features = np.concatenate((self.all_features, features.cpu().numpy()))
        self.n += n
        if self.n >= self.end_n:
            self.save_and_exit()

    def save_and_exit(self):
        import mmcv
        data = dict(count=self.count.cpu().numpy(), all_labels=self.all_labels)
        mmcv.dump(data, './mllt/appendix/VOCdevkit/longtail2012/resample_results_b6.pkl')
        print('resample result saved at :{}'.format('./mllt/appendix/VOCdevkit/longtail2012/resample_results_b6.pkl'))
        exit()


