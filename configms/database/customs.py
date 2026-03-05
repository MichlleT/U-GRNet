from __future__ import division,print_function,unicode_literals
import os,sys,cv2,glob
import albumentations as A
import numpy as np
from torch.utils.data import Dataset,DataLoader
from albumentations.core.transforms_interface import *
sys.path.append('../')


class ToTensorTest(BasicTransform):
    """Convert image and mask to `torch.Tensor`. The numpy `BHWC` image is converted to pytorch `BCHW` tensor.
    If the image is in `BHW` format (grayscale image), it will be converted to pytorch `BHW` tensor.
    Args:
        transpose_mask (bool): if True and an input mask has three dimensions, this transform will transpose dimensions
        so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
        standard format for PyTorch Tensors. Default: False.
    """

    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToTensorTest, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        if len(img.shape) not in [3, 4]:
            raise ValueError("Albumentations only supports images in BHW or BHWC format")

        if len(img.shape) == 3:
            img = np.expand_dims(img, 4)

        return torch.from_numpy(img.transpose(0, 3, 1, 2))

    def apply_to_mask(self, mask, **params):
        if self.transpose_mask and mask.ndim == 4:
            mask = mask.transpose(0, 3, 1, 2)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}




class CustomDataset(Dataset):
    def __init__(self,
                 img_dir,
                 sub_dir='A',
                 ann_dir=None,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 transform=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 size=256,
                 debug=False):
        self.transform = transform
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.sub_dir = sub_dir
        self.size = size
        self.debug = debug

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = os.path.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = os.path.join(self.data_root, self.ann_dir)

        self.img_infos = self.load_infos(self.img_dir, self.img_suffix,
                                         self.seg_map_suffix, self.sub_dir,
                                         self.ann_dir,self.split)

        if self.transform is None:
            self.transform = self.get_default_transform() if not self.test_mode \
                else self.get_test_transform()

        if self.debug:
            self.transform = A.Compose([t for t in self.transform if not isinstance(t, (A.Normalize, ToTensorV2,
                                                                                        ToTensorTest))])

    def load_infos(self, img_dir, img_suffix, seg_map_suffix, sub_dir,ann_dir, split):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name)
                    img_info['img'] = dict(img1_path=osp.join(img_dir, sub_dir, img_name))
                    if ann_dir is not None:
                        seg_map_path = os.path.join(ann_dir,
                                           img_name.replace(img_suffix, seg_map_suffix))
                        img_info['ann'] = dict(ann_path=seg_map_path)
                    img_infos.append(img_info)
        else:
            for img in glob.glob(os.path.join(img_dir, sub_dir, '*'+img_suffix)):
                img_name = os.path.basename(img)
                img_info = dict(filename=img_name)
                img_info['img'] = dict(img1_path=os.path.join(img_dir, sub_dir, img_name))
                if ann_dir is not None:
                    seg_map_path = os.path.join(ann_dir,
                                            img_name.replace(img_suffix, seg_map_suffix))
                    img_info['ann'] = dict(ann_path=seg_map_path)
                img_infos.append(img_info)

        print(f'Loaded {len(img_infos)} images')
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            A.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1)),     # div(255)
            ToTensorV2()
        ])
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""
        pass

    def get_image(self, img_info):
        # img1 = cv2.imread(img_info['img']['img1_path'])
        # print(img_info['img']['img1_path'])
        # img1 = cv2.cvtColor(cv2.imread(img_info['img']['img1_path']))
        # print(img1.shape)
        img1 = cv2.cvtColor(cv2.imread(img_info['img']['img1_path']), cv2.COLOR_BGR2RGB)
        return img1

    def get_gt_seg_maps(self, img_info, vis=False):
        # ann = cv2.imread(img_info['ann']['ann_path'], cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(img_info['ann']['ann_path'], 0)
        # ann = ann / 255 if not vis else ann
        return ann

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]

        img1 = self.get_image(img_info)
        return img1, img_info['filename']

    def prepare_img_ann(self, idx):
        img_info = self.img_infos[idx]
        img1 = self.get_image(img_info)
        ann = self.get_gt_seg_maps(img_info, self.debug)

        return img1, ann, img_info['filename']

    def format_results(self, results, **kwargs):
        """Place holder to format result to datasets specific output."""
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)


