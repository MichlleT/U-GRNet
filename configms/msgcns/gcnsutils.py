from __future__ import division,print_function,unicode_literals
import sys,os
sys.path.append('../')
import torch.nn as nn
from torch.nn import functional as F
import torch

from collections import OrderedDict
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)



class Basconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, is_batchnorm = False, kernel_size = 3, stride = 1, padding=1):
        super(Basconv, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.ReLU(inplace=True))

        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        return x



class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.05,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


class GloRe_Unit(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = Basconv(num_in, self.num_s, is_batchnorm = True, kernel_size=kernel_size, padding=padding)  
        # generate projection and inverse projection functions
        self.conv_proj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)   
        self.conv_reproj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)  
        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)   
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)  
        # fusion
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1), 
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in) 

    def forward(self, x):
        batch_size = x.size(0)
        # generate projection and inverse projection matrices
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        x_rproj_reshaped = self.conv_reproj(x).view(batch_size, self.num_n, -1)
        # project to node space
        x_n_state1 = torch.bmm(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1)) 
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        # graph convolution
        x_n_rel1 = self.gcn1(x_n_state2)  
        x_n_rel2 = self.gcn2(x_n_rel1)
        # inverse project to original space
        x_state_reshaped = torch.bmm(x_n_rel2, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        # fusion
        out = x + self.blocker(self.fc_2(x_state))

        return out


class  MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        # self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        # self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        # self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        # self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        # self.f1 = Basconv(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.f1 = Basconv(in_channels=2*out_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        # self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        # self.g2 = self.glou2(self.x2)
        # self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        # self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        # self.g3= self.glou3(self.x3)
        # self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        # out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)
        out = torch.cat([self.g0, self.layer1], 1)

        return self.f1(out)



class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_window=6, add_input=False):
        super(Attention_Embedding, self).__init__()
        self.add_input = add_input
        self.SE = nn.Sequential( 
            nn.AvgPool2d(kernel_size=pool_window+1, stride=1, padding = pool_window//2),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.BatchNorm2d(in_channels//reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels//reduction, out_channels, 1),
            nn.Sigmoid())
            
    def forward(self, high_feat, low_feat):
        b, c, h, w = low_feat.size()
        A = self.SE(high_feat)
        A = F.upsample(A, (h,w), mode='bilinear')
        
        output = low_feat*A
        if self.add_input:
            output += low_feat
        
        return output