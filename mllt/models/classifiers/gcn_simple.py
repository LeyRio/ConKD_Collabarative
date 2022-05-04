import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
import torch.nn.functional as F

@CLASSIFIERS.register_module
class GCNSimpleClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck1,
                 neck2,
                 head,
                 head2,
                 gcn1,
                 gcn2,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 savefeat=False,
                 t = None,
                 multilable_loss = None,
                 con_loss = None):
        super(GCNSimpleClassifier, self).__init__()
        # just resnet50 except last stage
        self.backbone = builder.build_backbone(backbone)
        # neck for main network:
        self.neck1 = builder.build_neck(neck1)
        self.neck2 = builder.build_neck(neck2)

        # mlp for GCN label:
        self.gcn1 = builder.build_head(gcn1)
        self.gcn2 = builder.build_head(gcn2)

        #head for contrastive:
        self.head1 = builder.build_head(head)
        self.head2 = builder.build_head(head2)
        self.multilabel_loss = builder.build_loss(multilable_loss)
        if con_loss !=None:
            self.con_loss = builder.build_loss(con_loss)

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
        super(GCNSimpleClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.gcn1.init_weights()
        self.gcn2.init_weights()
        self.neck1.init_weights(pretrained=pretrained)
        self.neck2.init_weights(pretrained=pretrained)
        self.head1.init_weights()
        self.head2.init_weights()


    # return self.forward_train(img, img_meta, **kwargs)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      flag,
                      epoches = None):
        if flag == 1:
            # data from CA loader,Need to split into two imgs:
            x = self.backbone(img)
            # 是否可以考虑和另一个分支的输出结果进行对比？
            #For main module:
            x_copy = x
            # 第一分支输出
            outs = self.neck1(x)
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
            #print(gt_labels.shape)
            loss_inputs2 = (outs2,gt_labels)
            # losses2 = self.gcn1.loss(*loss_inputs2)
            losses['gcn_losses'] = 0.1 * (self.multilabel_loss(outs2,gt_labels) + self.multilabel_loss(outs4,gt_labels))
            #losses['gcn_losses'] = 0.1 * (self.multilabel_loss(outs2,gt_labels))
            #losses['gcn_losses'] =losses2['loss_cls']
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
            loss_inputs3 = (outs_y2,gt_labels)
            losses = self.head2.loss(*loss_inputs)
            #losses2 = self.gcn2.loss(*loss_inputs3)
            #losses['gcn_losses'] = losses2['loss_cls']
            losses['gcn_losses'] = 0.1 * (self.multilabel_loss(outs_y2,gt_labels) + self.multilabel_loss(outs2_y2,gt_labels))
            #losses['gcn_losses'] = 0.1 * (self.multilabel_loss(outs_y2,gt_labels) )
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
        outs_1 = self.head1(outs)
        outs2 = self.neck2(y)
        outs_2 = self.head2(outs2)
        if isinstance(outs, tuple):
            outs = outs[-1]
        #print(x.shape)
        outs = F.avg_pool2d(outs, outs.size()[2:])
        outs = outs.view(outs.size(0), -1)
        if isinstance(outs2, tuple):
            outs2 = outs2[-1]
            #print(x.shape)
        outs2 = F.avg_pool2d(outs2, outs2.size()[2:])
        outs2 = outs2.view(outs2.size(0), -1)
        outs3 = self.gcn1(outs) 
        outs4 = self.gcn2(outs2)
        
        
        #outs3 = (outs + outs2)/2
        #outs5 = (outs_1 + outs_2 + outs3 + outs4)/4
        outs5 = (outs_1 + outs_2)/2
        outs6 = (outs3 + outs4)/2
        if self.savefeat:
            return outs5, x

        return outs5

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    def kd_test(self,img,img_metas,gt_labels,flag = 1,rescale = False):
        x = self.backbone(img)
        if flag == 1:
            outs = self.neck1(x)
            outs_1 = self.head1(outs)
        else:
            outs = self.neck2(x)
            outs_1 = self.head2(outs)
        return outs_1

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


