import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
import torch.nn.functional as F
import mmcv
import copy

# 任务：
# 1.理一遍整体训练的通路，看目前的情况
# 2.需要为collaborate_loss的参数修改为可训练参数
# 3.如何将Collaborate_loss和con_Colla_loss相加并训练？
# Add New file dualsimple 
@CLASSIFIERS.register_module
class DualKDClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 head,
                 head2,
                 neck1,
                 neck2,
                 neck=None,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 T = 1.0,
                 lamda = 10,
                 savefeat=False,
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 freq_file='./class_freq.pkl'):
        super(DualKDClassifier, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck1 is not None:
            self.neck1 = builder.build_neck(neck1)
        if neck2 is not None:
            self.neck2 = builder.build_neck(neck2)
        self.head = builder.build_head(head)
        self.head2 = builder.build_head(head2)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.lock_back=lock_back
        self.lock_neck=lock_neck
        self.savefeat=savefeat
        self.lamda = 1.0
        self.T = T
        self.lamda = lamda

        if self.savefeat and not self.with_neck:
            assert neck is not None, 'We must have a neck'
            assert train_cfg is None, 'this is only at testing stage'
        if self.lock_back:
            print('\033[1;35m >>> backbone locked !\033[0;0m')
        if self.lock_neck:
            print('\033[1;35m >>> neck locked !\033[0;0m')
        self.init_weights(pretrained=pretrained)
        self.count = CountMeter(num_classes=20)

        # Add New weights
        self.class_freq = torch.from_numpy(np.asarray(
            mmcv.load(freq_file)['class_freq'])).float().cuda()
        self.neg_class_freq = torch.from_numpy(
            np.asarray(mmcv.load(freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale
        #print(self.init_bias.shape)  [20]

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq
        self.rou = self.class_freq/self.train_num


    def init_weights(self, pretrained=None):
        super(DualKDClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck1.init_weights(pretrained=pretrained)
        self.neck2.init_weights(pretrained=pretrained)
        self.head.init_weights()
        self.head2.init_weights()

    def extract_feat(self, img):
        if self.lock_back:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)
        

        return x

    # TODO change loss
    # TODO 添加一个conloss的head
    # return self.forward_train(img, img_meta, **kwargs)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      soft,
                      flag,
                      epoches = None):
        x = self.extract_feat(img)
        weight = gt_labels * torch.exp(1-self.rou) + (1-gt_labels) * torch.exp(self.rou)
        y = x
        # 传入了某种分类的一个batch的数据，根据flag进行判断
        if flag == 1:
            # 复制一份传递给SubnetR
            x_copy = x
            # 第一分支输出
            #print("data1 for head1")
            outs = self.neck1(x)
            outs = self.head(outs)
            # 第二分支输出
            #print("data1 for head2")
            outs_copy = self.neck2(x_copy)
            outs2 = self.head2(outs_copy)
            #with open('test.txt','a') as file0:
            #    print(outs_copy,file=file0)
            """
            print("outs")
            print(outs_copy[0].shape)
            print(outs_copy[0][0][1])
            """
            
           
            # 得到loss,传入之后会计算cls_loss + con_loss
            loss_inputs1 = (outs, outs2, gt_labels)
            losses = self.head.loss(*loss_inputs1)
            losses['soft_losses'] = weight * self.lamda * F.kl_div(F.log_softmax(outs/self.T,dim=1),
                            F.softmax(soft/self.T,dim=1),reduction='batchmean')*self.T*self.T
        else:
            y_copy = y
            # 第二分支输出
            #print("data2 for head2")
            outs_y = self.neck2(y)
            outs_y = self.head2(outs_y)
            # 第一分支输出
            #print("data2 for head1")
            outs_copy_y = self.neck1(y_copy)
            outs2_y = self.head(outs_copy_y)

            loss_inputs = (outs_y,outs2_y,gt_labels)
            losses = self.head2.loss(*loss_inputs)
            losses['soft_losses'] =weight * self.lamda * F.kl_div(F.log_softmax(outs_y/self.T,dim=1),
                            F.softmax(soft/self.T,dim=1),reduction='batchmean')*self.T*self.T

        return losses

    # def simple_test(self, img, img_meta=None, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.head(x)
    #     return outs

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        y = x
        outs = self.neck1(x)
        outs = self.head(outs)
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


