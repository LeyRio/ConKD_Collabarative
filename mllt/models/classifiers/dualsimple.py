import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np

# 任务：
# 1.理一遍整体训练的通路，看目前的情况
# 2.需要为collaborate_loss的参数修改为可训练参数
# 3.如何将Collaborate_loss和con_Colla_loss相加并训练？
# Add New file dualsimple 
@CLASSIFIERS.register_module
class DualSimpleClassifier(BaseClassifier):

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
                 savefeat=False):
        super(DualSimpleClassifier, self).__init__()
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

        if self.savefeat and not self.with_neck:
            assert neck is not None, 'We must have a neck'
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
        super(DualSimpleClassifier, self).init_weights(pretrained)
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

        """
        if self.with_neck:
            if self.lock_neck:
                with torch.no_grad():
                    x = self.neck(x)
            else:
                x = self.neck(x)
        """
        

        return x

    # TODO change loss
    # TODO 添加一个conloss的head
    # return self.forward_train(img, img_meta, **kwargs)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      flag,
                      epoches = None):
        x = self.extract_feat(img)
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


