#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division,print_function,unicode_literals
import os,sys,random
import numpy as np
from scipy import stats
import torch.nn as nn
sys.path.append('../')
from utils import base

Classes = ['0','1','2','3','4','5']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(pred, label):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def Confusion_matrix_res(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def Evaluate_res(conf_mat):
    acc = np.diag(conf_mat).sum() / conf_mat.sum()
    acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_per_class)

    IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_IoU = np.nanmean(IoU)

    # 求kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
    kappa = (acc - pe) / (1 - pe)

    ## p,r,f-score

    precision = np.diag(conf_mat) / conf_mat.sum(axis = 1)
    recall = np.diag(conf_mat) / conf_mat.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    mean_fs = np.nanmean(f1score)

    freq = np.sum(conf_mat,axis=1) / np.sum(conf_mat)
    fwiou = (freq[freq > 0] * IoU[freq > 0]).sum()

    return acc, mean_IoU, kappa,fwiou,mean_fs



def Evaluates(conf_mat):
    acc = np.diag(conf_mat).sum() / conf_mat.sum()
    acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_per_class)

    IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_IoU = np.nanmean(IoU)

    # 求kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
    kappa = (acc - pe) / (1 - pe)

    ## p,r,f-score

    precision = np.diag(conf_mat) / conf_mat.sum(axis = 1)
    recall = np.diag(conf_mat) / conf_mat.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    mean_fs = np.nanmean(f1score)

    freq = np.sum(conf_mat,axis=1) / np.sum(conf_mat)
    fwiou = (freq[freq > 0] * IoU[freq > 0]).sum()

    dice = (2 * np.diag(conf_mat)) / (conf_mat.sum(axis = 1) + conf_mat.sum(axis=0))
    dice = np.nanmean(dice)
    sen = np.diag(conf_mat) / conf_mat.sum(axis=0)
    sen = np.nanmean(sen)

    spe = (np.diag(conf_mat).sum() - np.diag(conf_mat)) / (np.diag(conf_mat).sum() + conf_mat.sum(axis=0))
    spe = np.nanmean(spe)

    return acc, mean_IoU, dice, sen, spe
