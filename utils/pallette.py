#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division,print_function,unicode_literals
import os,sys,random
import numpy as np

# COLORMAP = [[0, 0, 0],[172, 214, 255], [153, 255, 51],
#                 [255, 230, 111], [255, 117, 117], [211, 164, 255]]
COLORMAP = [[0, 0, 0],[255, 214, 172], [51, 255, 153],
                [111, 230, 255], [255, 164, 211], [117, 117, 255]]

def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


# def get_random_color():
#     """获取一个随机的颜色"""
#     c = []
#     for index in range(13):
#         r = lambda: random.randint(0,255)
#         c.append([r(),r(),r()])
#     return c

# c = get_random_color()
# print(c)
