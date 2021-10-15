# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn.functional as F
from .backbones.resnet_ibn_a import resnet50_ibn_a

class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck_planes, dropout_rate, model_name,pretrain_choice,gem_pooling,id_loss_type,cfg):
        super(Baseline, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        self.gem_pooling = gem_pooling
        if self.gem_pooling:
            print('using GeM pooling')
            self.gem = GeM()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck_planes = neck_planes
        self.dropout_rate = dropout_rate

        if self.neck_planes > 0:
            self.fcneck = nn.Linear(self.in_planes, self.neck_planes, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.fcneck_bn = nn.BatchNorm1d(self.neck_planes)
            self.fcneck_bn.apply(weights_init_kaiming)
            self.in_planes = self.neck_planes
            #print('fcneck is used.')

            self.relu = nn.ReLU(inplace=True)
        else:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            #print('dropout is used: %f.' %self.dropout_rate)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x,label=None):

        x = self.base(x)
        if self.gem_pooling:
            global_feat = self.gem(x)
        else:
            global_feat = self.gap(x) + self.gmp(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], global_feat.shape[1])  # flatten to (bs, 2048)
        # global_feat = global_feat.view(1, 2048)  # flatten to (bs, 2048)

        if self.neck_planes > 0:
            global_feat = self.fcneck(global_feat)
            global_feat = self.fcneck_bn(global_feat)
            #global_feat = self.relu(global_feat)
            if self.dropout_rate > 0:
                global_feat_cls = self.dropout(global_feat)
            else:
                global_feat_cls = global_feat

            if self.training:
                cls_score = self.classifier(global_feat_cls)
                return cls_score, global_feat  # global feature for triplet loss
            else:
                return global_feat
        else:
            feat = self.bottleneck(global_feat)
            if self.training:
                cls_score = self.classifier(feat)
                return cls_score, global_feat
            else:
                return feat  

    def load_param(self, trained_path, pretrain_choice):
        if pretrain_choice == 'self':
            param_dict = torch.load(trained_path, map_location='cpu')
            for i in param_dict:
                if 'classifier' in i:
                    continue
                self.state_dict()[i].copy_(param_dict[i])
        elif pretrain_choice == 'imagenet':
            self.base.load_param(trained_path)
