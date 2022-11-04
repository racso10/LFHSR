import torch
import numpy as np
import torch.utils.data as data
import os
import utils_train
import math


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.output_size = (output_size, output_size)

    def __call__(self, train_data, gt_image, gt_mask_image, scale, shave=0):
        h, w = train_data.shape[-2], train_data.shape[-1]
        new_h, new_w = self.output_size

        top = np.random.randint(0 + shave, h - new_h - shave)
        left = np.random.randint(0 + shave, w - new_w - shave)

        train_data_tmp = train_data[..., top: top + new_h, left: left + new_w]

        gt_data_tmp = gt_image[..., top * scale: top * scale + new_h * scale,
                      left * scale: left * scale + new_w * scale]

        if gt_mask_image is None:
            gt_mask_image = None
        else:
            gt_mask_image = gt_mask_image[..., top * scale: top * scale + new_h * scale,
                            left * scale: left * scale + new_w * scale]

        return train_data_tmp, gt_data_tmp, gt_mask_image


class LFHSR_shear_Dataset(data.Dataset):
    """Light Field dataset."""

    def __init__(self, dir_LF, repeat=4, view_n_ori=9, view_n_input=9, scale=2, disparity_list=None, crop_size=32,
                 view_position=None, if_flip=False, if_rotation=False, is_all=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.crop_size = crop_size
        self.repeat = repeat
        self.view_n_ori = view_n_ori
        self.view_n_input = view_n_input
        self.RandomCrop = RandomCrop(crop_size)
        self.if_flip = if_flip
        self.if_rotation = if_rotation
        self.is_all = is_all
        self.scale = scale
        self.dir_LF = dir_LF
        self.train_data_all = []
        self.gt_data_all = []
        # self.gt_mask_data_all = []
        self.gt_mask_data_all = []
        self.numbers = len(os.listdir(dir_LF))
        self.view_position = view_position
        self.D = len(disparity_list)

        img_list = os.listdir(dir_LF)
        img_list.sort()
        for img_name in img_list:
            train_data, gt_data, gt_mask_data = utils_train.data_prepare_new(dir_LF + img_name, view_n_ori, view_n_input,
                                                                            scale, disparity_list)
            self.train_data_all.append(train_data)
            self.gt_data_all.append(gt_data)
            self.gt_mask_data_all.append(gt_mask_data)

        self.shave = math.ceil(max(disparity_list) * (view_n_input - 1) // 2 / scale)

    def __len__(self):
        return self.repeat * self.numbers

    def __getitem__(self, idx):

        train_data = self.train_data_all[idx // self.repeat]
        gt_data = self.gt_data_all[idx // self.repeat]
        gt_mask_data = self.gt_mask_data_all[idx // self.repeat]

        train_data, gt_data, gt_mask_data = self.RandomCrop(train_data, gt_data, gt_mask_data, self.scale, self.shave)

        if self.if_flip:
            if np.random.rand(1) >= 0.5:
                train_data = np.flip(train_data, 3)
                train_data = np.flip(train_data, 1)

                gt_data = np.flip(gt_data, 2)
                gt_data = np.flip(gt_data, 0)

                if gt_mask_data is not None:
                    gt_mask_data = np.flip(gt_mask_data, 1)

        if self.if_rotation:
            k = np.random.randint(0, 4)
            train_data = np.rot90(train_data, k, (3, 4))
            train_data = np.rot90(train_data, k, (1, 2))

            gt_data = np.rot90(gt_data, k, (2, 3))
            gt_data = np.rot90(gt_data, k, (0, 1))

            if gt_mask_data is not None:
                gt_mask_data = np.rot90(gt_mask_data, k, (1, 2))

        if gt_mask_data is None:
            gt_mask_data = np.array([-1])

        if self.is_all:
            return torch.from_numpy(train_data.copy()), \
                   torch.from_numpy(train_data[self.D // 2].copy()), \
                   torch.from_numpy(gt_data[self.view_n_input // 2, self.view_n_input // 2].copy()), -1, \
                   torch.from_numpy(gt_data.copy()), \
                   torch.from_numpy(gt_mask_data.copy())
        else:
            view_u = np.random.randint(0, self.view_n_input)
            view_v = np.random.randint(0, self.view_n_input)
            view_position = (view_u, view_v)

            return torch.from_numpy(train_data.copy()), \
                   torch.from_numpy(train_data[self.D // 2, view_u, view_v].copy()), \
                   torch.from_numpy(gt_data[self.view_n_input // 2, self.view_n_input // 2].copy()), \
                   view_position, \
                   torch.from_numpy(gt_data[view_u, view_v].copy())
