import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np

@CLASSIFIERS.register_module
class ContrastSimpleClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck1,
                 neck2,
                 head,
                 head2,
                 mlp1,
                 mlp2,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 savefeat=False,
                 dim = 128,
                 K = 65536,
                 m = 0.999,
                 contrast_loss = None,
                 classes = 20,
                 tempreture = 0,
                 base_tempreture = 0):
        super(ContrastSimpleClassifier, self).__init__()
        self.m = m
        self.K = K
        self.T = tempreture
        self.BT = base_tempreture
        # just resnet50 except last stage
        self.backbone = builder.build_backbone(backbone)
        #backbone for contrastive:
        self.encoder_k = builder.build_backbone(backbone)
        # neck for main network:
        self.neck1 = builder.build_neck(neck1)
        self.neck2 = builder.build_neck(neck2)
        #neck for contrastive:
        self.encoder_neck_k = builder.build_neck(neck1) 

        # mlp for contrastive:
        self.mlp_q = builder.build_head(mlp1)
        self.mlp_k = builder.build_head(mlp2)

        #head for contrastive:
        self.head1 = builder.build_head(head)
        self.head2 = builder.build_head(head2)
        self.contrastloss = builder.build_loss(contrast_loss)

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # type(queue) : class 'torch.Tensor'
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_label", torch.zeros(classes, K))
        #self.queue_label = nn.functional.normalize(self.queue_label, dim=0)

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
        super(ContrastSimpleClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.mlp_q.init_weights()
        self.neck1.init_weights(pretrained=pretrained)
        self.neck2.init_weights(pretrained=pretrained)
        self.head1.init_weights()
        self.head2.init_weights()
        for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.neck1.parameters(), self.encoder_neck_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    # return self.forward_train(img, img_meta, **kwargs)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      flag):
        if flag == 1:
            # data from CA loader,Need to split into two imgs:
            img1 = img[0]
            img2 = img[1]
            gt_labels = gt_labels[0]
            #print(type(gt_labels))
            # first under backbone:
            x1 = self.backbone(img1)

            # For contrastive:
            x = self.neck1(x1)
            feat1 = self.mlp_q(x)
            query = nn.functional.normalize(feat1, dim=1)
            with torch.no_grad(): 
                self._momentum_update_key_encoder()  # update the key encoder
                x2 = self.encoder_k(img2)
                x2 = self.encoder_neck_k(x2)
                feat2 = self.mlp_k(x2)
                keys = nn.functional.normalize(feat2, dim=1)
            #print(type(gt_labels))
            self.queue_dequeue(keys,gt_labels)

            #For main module:
            x_copy = x1
            # 第一分支输出
            outs = self.neck1(x1)
            outs = self.head1(outs)
            # 第二分支输出
            outs_copy = self.neck2(x_copy)
            outs2 = self.head2(outs_copy)
            # 得到loss,传入之后会计算cls_loss + con_loss
            loss_inputs1 = (outs, outs2, gt_labels)
            losses = self.head1.loss(*loss_inputs1)
            # Add contrastive loss:
            losses['loss_contrastive'] = 0.1 *self.contrastloss(query,keys,gt_labels,self.queue,self.queue_label,self.K,self.T,self.BT) 
        else:
            #data from Random loader,just one picture
            y = self.backbone(img)
            with torch.no_grad(): 
                self._momentum_update_key_encoder()  # update the key encoder
            y_copy = y
            # 第二分支输出
            #print("data2 for head2")
            outs_y = self.neck2(y)
            outs_y = self.head2(outs_y)
            # 第一分支输出
            #print("data2 for head1")
            outs_copy_y = self.neck1(y_copy)
            outs2_y = self.head1(outs_copy_y)

            loss_inputs = (outs_y,outs2_y,gt_labels)
            losses = self.head2.loss(*loss_inputs)
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

    # need to check if it is right
    def queue_dequeue(self,keys,labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[:,ptr:ptr + batch_size] = labels.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.neck1.parameters(), self.encoder_neck_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
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


