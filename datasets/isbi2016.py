import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


ISBI2016_CLASS_NAMES = ['lesion']


def _mask_path_from_image_path(mask_dir, image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(mask_dir, f'{base_name}_Segmentation.png')


def _collect_image_mask_pairs(image_dir, mask_dir):
    image_paths = sorted(
        os.path.join(image_dir, file_name)
        for file_name in os.listdir(image_dir)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    x, y, mask, img_types = [], [], [], []
    for image_path in image_paths:
        mask_path = _mask_path_from_image_path(mask_dir, image_path)
        if not os.path.exists(mask_path):
            continue
        x.append(image_path)
        y.append(1)
        mask.append(mask_path)
        img_types.append('lesion')

    return x, y, mask, img_types


class ISBI2016Dataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in ISBI2016_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, ISBI2016_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size

        if is_train:
            image_dir = os.path.join(self.dataset_path, 'ISBI2016_ISIC_Part1_Training_Data')
            mask_dir = os.path.join(self.dataset_path, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
        else:
            image_dir = os.path.join(self.dataset_path, 'ISBI2016_ISIC_Part1_Test_Data')
            mask_dir = os.path.join(self.dataset_path, 'ISBI2016_ISIC_Part1_Test_GroundTruth')

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f'Cannot find ISBI2016 image directory: {image_dir}')
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f'Cannot find ISBI2016 mask directory: {mask_dir}')

        self.x, self.y, self.mask, self.img_types = _collect_image_mask_pairs(image_dir, mask_dir)

        if excluded_images is not None:
            excluded = {os.path.basename(path) for path in excluded_images}
            keep_flags = [os.path.basename(path) not in excluded for path in self.x]
            self.x = [path for path, flag in zip(self.x, keep_flags) if flag]
            self.y = [value for value, flag in zip(self.y, keep_flags) if flag]
            self.mask = [value for value, flag in zip(self.mask, keep_flags) if flag]
            self.img_types = [value for value, flag in zip(self.img_types, keep_flags) if flag]

        self.transform_x = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor(),
        ])
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor(),
        ])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask_path, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]

        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))

        mask = Image.open(mask_path).convert('L')
        mask = self.transform_mask(mask)
        mask = (mask > 0).float()

        file_name = os.path.splitext(os.path.basename(img_path))[0]
        return x, y, mask, file_name, img_type

    def __len__(self):
        return len(self.x)
