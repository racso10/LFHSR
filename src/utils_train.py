import numpy as np
import cv2
import skimage.color as color
import time
import torch
from PIL import Image
import torch.nn.functional as F
import math


def weight_L1Loss(X, Y, weight):
    weight_error = torch.abs(X - Y) * weight
    loss = torch.sum(weight_error) / torch.numel(weight_error)
    return loss


def warp_all(img, disparity_list, view_n, view_position, align_corners=False):
    B, C, UV, X, Y = list(img.shape)
    D = len(disparity_list)
    # img = img.unsqueeze(0)
    # img = img.expand(D, -1, -1, -1, -1, -1)  # D, B, C, UV, X, Y
    img = img.permute(2, 0, 1, 3, 4).reshape(UV, B * C, X, Y)  # DUV ,B, C, X, Y
    # view_central = view_n // 2
    img_all = []
    target_position = np.array([view_position[0], view_position[1]])  #

    for disparity in disparity_list:
        theta = []
        for i in range(view_n):
            for j in range(view_n):
                ref_position = np.array([i, j])
                d = (target_position - ref_position) * disparity * 2
                theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]])
                theta.append(theta_t.unsqueeze(0))
        theta = torch.cat(theta, 0).cuda()
        grid = F.affine_grid(theta, img.size(), align_corners=align_corners)
        img_tmp = F.grid_sample(img, grid, align_corners=align_corners)
        img_tmp.unsqueeze(0)
        img_all.append(img_tmp)
    img_all = torch.cat(img_all, 0)
    img_all = img_all.reshape(D, UV, B, C, X, Y).permute(2, 3, 0, 1, 4, 5)  # B, C, D, UV, X, Y
    return img_all


def data_prepare_new(dir_LF, view_n_ori, view_n_new, scale, disparity_list):
    assert view_n_new % 2 == 1
    D = len(disparity_list)
    gt_y = image_prepare_npy(dir_LF, view_n_ori, view_n_new)
    U, V, X, Y = list(gt_y.shape)
    lr_X = X // scale
    lr_Y = Y // scale
    X = lr_X * scale
    Y = lr_Y * scale
    lr_y = np.zeros((view_n_new, view_n_new, lr_X, lr_X), dtype=np.float32)
    gt_y = gt_y[..., :X, :Y]

    for i in range(view_n_new):
        for j in range(view_n_new):
            img = Image.fromarray(gt_y[i, j] / 255.0)
            img_tmp = img.resize((lr_X, lr_Y), Image.BICUBIC)
            lr_y[i, j, ...] = img_tmp
    lr_y = torch.from_numpy(lr_y.copy()).cuda().reshape(1, 1, -1, lr_X, lr_Y)
    lr_y_sheared = warp_all(lr_y, disparity_list / scale, view_n_new, view_position=[view_n_new // 2, view_n_new // 2])
    lr_y_sheared = lr_y_sheared.reshape(D, U, V, lr_X, lr_Y).cpu().numpy()
    gt_y /= 255.0

    return lr_y_sheared, gt_y, None


def angular_resolution_changes(image, view_num_ori, view_num_new):
    n_view = (view_num_ori + 1 - view_num_new) // 2
    return image[n_view:n_view + view_num_new, n_view:n_view + view_num_new, :, :]


def image_prepare_npy(image_path, view_n_ori, view_n_new):
    # gt_image = np.load(image_path).astype(np.float32)
    gt_image = np.load(image_path)
    gt_image = gt_image.astype(np.float32)
    # change the angular resolution of LF images for different input
    if view_n_new < view_n_ori:
        gt_image_input = angular_resolution_changes(gt_image, view_n_ori, view_n_new)
    else:
        gt_image_input = gt_image

    return gt_image_input


def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
