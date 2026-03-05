#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division,print_function,unicode_literals
import os,sys,torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append('../')
from utils import base
from utils.dice import *



class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class MultiHeadCELoss(torch.nn.Module):
    __name__ = "MultiHeadCELoss"

    def __init__(self,
                 index_weight=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                 weight=None,
                 reduction='mean',
                 loss2=False,
                 loss2_weight=1.0,
                 **kwargs
                 ):
        super(MultiHeadCELoss, self).__init__()
        self.index_weight = index_weight
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss_functions = [CrossEntropyLoss(weight=weight, reduction=reduction, **kwargs)
                               for _ in index_weight]
        if self.loss2:
            self.loss_functions_dice = [DiceLoss(mode='multiclass') for _ in index_weight]

    def forward(self, preds, target):
        losses = []

        for i in range(len(preds)):
            scale_ = preds[i].shape[-1]

            if target.size()[2] == scale_:
                scale_target = target
            else:
                tmp = torch.unsqueeze(target, dim=1)
                tmp = torch.nn.functional.interpolate(tmp.float(), size=[scale_, scale_])
                scale_target = torch.squeeze(tmp, dim=1)
            loss = self.loss_functions[i](preds[i], scale_target.long())

            if self.loss2:
                loss_dice = self.loss_functions_dice[i](preds[i], scale_target.long())
                losses.append(self.index_weight[i] * (loss + loss_dice * self.loss2_weight))
            else:
                losses.append(self.index_weight[i] * loss)

        return sum(losses)
        # return losses[0] + losses[1] + losses[2] + losses[3] + losses[4]



class Multi_MultiHeadCELoss(torch.nn.Module):
    __name__ = "Multi_MultiHeadCELoss"

    def __init__(self,
                 index_weight=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                 weight=None,
                 reduction='mean',
                 loss2=False,
                 loss2_weight=1.0,
                 **kwargs
                 ):
        super(Multi_MultiHeadCELoss, self).__init__()
        self.index_weight = index_weight
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss_functions = [CrossEntropyLoss(weight=weight, reduction=reduction, **kwargs)
                               for _ in index_weight]
        if self.loss2:
            self.loss_functions_dice = [DiceLoss(mode='multiclass') for _ in index_weight]

    def forward(self, preds, target):
        losses = []

        for i in range(len(preds)):
            for j in range(2):
                scale_ = preds[i][j].shape[-1]

                if target.size()[2] == scale_:
                    scale_target = target
                else:
                    tmp = torch.unsqueeze(target, dim=1)
                    tmp = torch.nn.functional.interpolate(tmp.float(), size=[scale_, scale_])
                    scale_target = torch.squeeze(tmp, dim=1)
                loss = self.loss_functions[i](preds[i][j], scale_target.long())

                if self.loss2:
                    loss_dice = self.loss_functions_dice[i](preds[i][j], scale_target.long())
                    losses.append(self.index_weight[i] * (loss + loss_dice * self.loss2_weight))
                else:
                    losses.append(self.index_weight[i] * loss)

        return sum(losses)