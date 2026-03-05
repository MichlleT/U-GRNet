#!/usr/bin/env python
from __future__ import division,print_function,unicode_literals
import os,sys,cv2,mmcv,argparse
sys.path.append('../')
from skimage import io
from tqdm import tqdm
import numpy as np

COLORMAP = [[0, 0, 0],[255, 255, 255],[0, 0, 255],[0, 255, 255],
            [0, 255, 0], [255, 255, 0], [255, 0, 0],[14, 133, 232],
            [182, 222, 131], [49, 166, 227], [169, 232, 11],
             [106, 118, 147],[255, 59, 96], [161, 1, 116],
              [212, 78, 22], [195, 146, 129], [217, 124, 114],
               [170, 54, 33], [111, 39, 249], [18, 236, 29]]

def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def makedirs_func(source):
	if not os.path.exists(source):
		os.makedirs(source)
	else:
		pass

def parse_args():
	parser = argparse.ArgumentParser(description='vis imgs')
	parser.add_argument('-i',dest='inputs',type=str,default='../datasets/inputs/Task01/samples/test/mask')
	parser.add_argument('-o',dest='outputs',type=str,default='../datasets/outputs/infors/Task01/vis')
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	makedirs_func(args.outputs)
	for names in tqdm(os.listdir(args.inputs)):
		imgs = io.imread(os.path.join(args.inputs,names)).astype(np.uint8)
		masks = Index2Color(imgs)
		# cv2.imwrite(os.path.join(args.outputs,names),imgs)
		io.imsave(os.path.join(args.outputs,names),masks)
		print(masks.shape)


if __name__ == "__main__":
	main()


