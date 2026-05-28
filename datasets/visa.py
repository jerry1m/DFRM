import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import excluding_images


if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


VISA_CLASS_NAMES = [
    'candle',
    'capsules',
    'cashew',
    'chewinggum',
    'fryum',
    'macaroni1',
    'macaroni2',
    'pcb1',
    'pcb2',
    'pcb3',
    'pcb4',
    'pipe_fryum',
]


def _resolve_rel_path(root_path, rel_path):
    rel = (rel_path or '').replace('\\', '/')
    return os.path.join(root_path, rel)


def _read_visa_split_csv(dataset_path):
    csv_path = os.path.join(dataset_path, 'split_csv', '1cls.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            'Cannot find VisA split file: {}. '
            'Expected official VisA layout with split_csv/1cls.csv.'.format(csv_path)
        )

    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


class VisADataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size

        if excluded_images is not None:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
            self.x, self.y, self.mask, self.img_types = excluding_images(self.x, self.y, self.mask, self.img_types, excluded_images)
        else:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()

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
        img_path, y, mask, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]

        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))

        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            mask = (mask > 0).float()

        file_name = os.path.splitext(os.path.basename(img_path))[0]
        return x, y, mask, file_name, img_type

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        split_name = 'train' if self.is_train else 'test'
        rows = _read_visa_split_csv(self.dataset_path)

        x, y, mask, types = [], [], [], []
        for row in rows:
            if row.get('object') != self.class_name:
                continue
            if row.get('split') != split_name:
                continue

            label = (row.get('label') or '').strip().lower()
            img_path = _resolve_rel_path(self.dataset_path, row.get('image'))
            if not os.path.exists(img_path):
                continue

            if label == 'normal':
                x.append(img_path)
                y.append(0)
                mask.append(None)
                types.append('normal')
            else:
                mask_rel = row.get('mask')
                mask_path = _resolve_rel_path(self.dataset_path, mask_rel) if mask_rel else None
                if mask_path is None or not os.path.exists(mask_path):
                    continue

                x.append(img_path)
                y.append(1)
                mask.append(mask_path)
                anomaly_type = row.get('anomaly_type')
                if not anomaly_type:
                    anomaly_type = os.path.basename(os.path.dirname(img_path))
                types.append(anomaly_type)

        return list(x), list(y), list(mask), list(types)


class VisAFSDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.normal_nums = 'all'
        self.reuse_times = 5

        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.reuse_times
        self.a_labels = self.a_labels * self.reuse_times
        self.a_masks = self.a_masks * self.reuse_times

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
        if idx >= len(self.n_imgs):
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]

        x = Image.open(img).convert('RGB')
        x = self.normalize(self.transform_x(x))

        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            mask = (mask > 0).float()

        return x, label, mask

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def load_dataset_folder(self):
        rows = _read_visa_split_csv(self.dataset_path)

        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []

        anomaly_pool = {}
        for row in rows:
            if row.get('object') != self.class_name:
                continue

            split_name = row.get('split')
            label = (row.get('label') or '').strip().lower()
            img_path = _resolve_rel_path(self.dataset_path, row.get('image'))
            if not os.path.exists(img_path):
                continue

            if split_name == 'train' and label == 'normal':
                n_img_paths.append(img_path)
                n_labels.append(0)
                n_mask_paths.append(None)
            elif split_name == 'test' and label != 'normal':
                mask_rel = row.get('mask')
                mask_path = _resolve_rel_path(self.dataset_path, mask_rel) if mask_rel else None
                if mask_path is None or not os.path.exists(mask_path):
                    continue

                anomaly_type = row.get('anomaly_type')
                if not anomaly_type:
                    anomaly_type = os.path.basename(os.path.dirname(img_path))
                anomaly_pool.setdefault(anomaly_type, []).append((img_path, mask_path))

        for anomaly_type, pairs in anomaly_pool.items():
            random.shuffle(pairs)
            selected = pairs[:min(self.anomaly_nums, len(pairs))]
            for img_path, mask_path in selected:
                a_img_paths.append(img_path)
                a_labels.append(1)
                a_mask_paths.append(mask_path)

        if self.normal_nums != 'all':
            random.shuffle(n_img_paths)
            n_img_paths = n_img_paths[:self.normal_nums]
            n_labels = n_labels[:self.normal_nums]
            n_mask_paths = n_mask_paths[:self.normal_nums]

        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisAFSCopyPasteDataset(VisAFSDataset):
    """
    Lightweight VisA FAS dataset.
    For compatibility with strategy '0,1', this currently reuses only real anomalies.
    """

    pass