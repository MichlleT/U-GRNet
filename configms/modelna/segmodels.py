from __future__ import division,print_function,unicode_literals
import os,sys
sys.path.append('../')
from configms.msgcns.model import *


def SegModels(args, mode):
    if mode == 'gcnpps':
        model = GUnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', 
            in_channels=3, classes=args.numclasses,  decoder_attention_type='scse') # decoder_attention_type='scse'
    else:
        pass

    return model
