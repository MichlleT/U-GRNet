from __future__ import division,print_function,unicode_literals
import os,sys
from tqdm import tqdm
sys.path.append('../')
from albumentations.pytorch import ToTensorV2
from .customs import *


def makedirs_func(source):
    if not os.path.exists(source):
        os.makedirs(source)
    else:
        print('the source is exists!')


def weightsAc(train_loader,num_classes):
    num_pixel = np.zeros((1, num_classes)).squeeze(0).astype(np.uint64)
    for index, data in enumerate(tqdm(train_loader)):
        image, mask, names = data
        height, weight = mask.shape[1], mask.shape[2]
        mask = mask.numpy().reshape(height * weight * mask.shape[0], 1).squeeze(1).astype(np.uint8)
        num_pixel_per_batch = np.bincount(mask, minlength=num_classes)
        num_pixel = num_pixel_per_batch + num_pixel

    proportion_pixel = num_pixel / num_pixel.sum()
    weight_temp = 1 / proportion_pixel
    max_in_weight = max(weight_temp)
    weight = weight_temp / max_in_weight


    return list(weight)


class Medical_Dataset(CustomDataset):
    def __init__(self, img_dir, sub_dir='images', ann_dir=None, img_suffix='.png',
                 seg_map_suffix='.png',
                 transform=None, split=None, data_root=None, test_mode=False, size=512, debug=False):
        super().__init__(img_dir, sub_dir, ann_dir, img_suffix, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    def get_default_transform(self):
        transform = A.Compose([
            A.RandomResizedCrop(self.size, self.size, p=0.8),
            A.OneOf([
                A.GaussNoise(p=1),
                A.Blur(p=1),
                A.IAASharpen(p=1),
                A.ISONoise(p=1),
                A.MotionBlur(p=1),
                A.IAAEmboss(),
                A.CLAHE(),
                # A.RandomBrightnessContrast(p=1),
                A.NoOp(p=1),
            ], p=1),

            A.OneOf([
                A.RandomGridShuffle(grid=(2, 2), p=1),
                # Mosaic(256, self.img_infos, self.__len__(), p=1),
                A.NoOp(p=1),
            ], p=0.3),

            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'images': 'image'})

        return transform

    def get_test_transform(self):
        test_transform = A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ], additional_targets={'images': 'image'})
        return test_transform

    def __getitem__(self, idx):
        if not self.ann_dir:
            img1, filename = self.prepare_img(idx)
            transformed_data = self.transform(image=img1)
            img1 = transformed_data['image']
            return img1, filename

        else:
            img1, ann, filename = self.prepare_img_ann(idx)
            transformed_data = self.transform(image=img1, mask=ann)
            img1, ann = transformed_data['image'], transformed_data['mask']

        return img1, ann, filename
