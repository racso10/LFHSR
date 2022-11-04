import numpy as np
import cv2
import skimage.color as color
import time
import torch
from PIL import Image
import torch.nn.functional as F
import math


def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def overlap_crop_forward(lr_LF_shear, hr_img, scale, disparity_list, model, max_length=32, shave=10, mod=16):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    b, d, uv, h, w = lr_LF_shear.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = int(math.ceil((h_half + shave) / mod) * mod), int(math.ceil((w_half + shave) / mod) * mod)
    lr_list = [
        lr_LF_shear[..., 0:h_size, 0:w_size],
        lr_LF_shear[..., 0:h_size, (w - w_size):w],
        lr_LF_shear[..., (h - h_size):h, 0:w_size],
        lr_LF_shear[..., (h - h_size):h, (w - w_size):w]]
    hr_list = [
        hr_img[:, :, 0:h_size * scale, 0:w_size * scale],
        hr_img[:, :, 0:h_size * scale, (w - w_size) * scale:w * scale],
        hr_img[:, :, (h - h_size) * scale:h * scale, 0:w_size * scale],
        hr_img[:, :, (h - h_size) * scale:h * scale, (w - w_size) * scale:w * scale]]

    sr_list = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        hr_batch = torch.cat(hr_list[i:(i + n_GPUs)], dim=0)
        if lr_batch.shape[3] > max_length or lr_batch.shape[4] > max_length:
            sr_batch_temp = overlap_crop_forward(lr_batch, hr_batch, scale, disparity_list, model,
                                                 max_length=max_length, shave=shave)
        else:
            lr_LF_batch = lr_batch[:, len(disparity_list) // 2]
            lr_LF_batch = lr_LF_batch.reshape(b, int(math.sqrt(uv)), int(math.sqrt(uv)), h_size, w_size)
            sr_batch_temp = model(lr_batch, lr_LF_batch, hr_batch, disparity_list)

        if isinstance(sr_batch_temp, list):
            sr_batch = sr_batch_temp[-1]
        else:
            sr_batch = sr_batch_temp

        sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))

    h, w = scale * h, scale * w
    h_half, w_half = h // 2, w // 2
    shave *= scale

    output = lr_LF_shear.new(b, 1, uv, h, w)

    output[..., :h_half, :w_half] = sr_list[0][..., :h_half, :w_half]
    output[..., :h_half, w_half:] = sr_list[1][..., :h_half, -w_half:]
    output[..., h_half:, :w_half] = sr_list[2][..., -h_half:, :w_half]
    output[..., h_half:, w_half:] = sr_list[3][..., -h_half:, -w_half:]

    return output


def warp_all(img, disparity_list, view_n, view_position, align_corners=False):
    B, C, UV, X, Y = list(img.shape)
    D = len(disparity_list)
    img = img.permute(2, 0, 1, 3, 4).reshape(UV, B * C, X, Y)  # DUV ,B, C, X, Y
    img_all = []
    target_position = np.array([view_position[0], view_position[1]])

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


def image_input(image_path, scale, view_ori, view_n):
    gt_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:, :, 0:3]
    gt_image = gt_image[:, :, ::-1]
    gt_image_ycbcr = color.rgb2ycbcr(gt_image).astype(np.float32)

    num_vew_gap = (view_ori + 1 - view_n) // 2

    image_h = gt_image_ycbcr.shape[0] // view_ori
    image_w = gt_image_ycbcr.shape[1] // view_ori
    channel_n = gt_image_ycbcr.shape[2]

    if image_h % 8 != 0:
        gt_image_ycbcr = gt_image_ycbcr[:-(image_h % 8) * view_ori, :, :]
        image_h -= image_h % 8
    if image_w % 8 != 0:
        gt_image_ycbcr = gt_image_ycbcr[:, :-(image_w % 8) * view_ori, :]
        image_w -= image_w % 8

    gt_ycbcr = np.zeros((1, view_n, view_n, image_h, image_w, channel_n), dtype=np.float32)
    lr_ycbcr = np.zeros((1, view_n, view_n, image_h // scale, image_w // scale, channel_n), dtype=np.float32)

    for i in range(0, view_n, 1):
        for j in range(0, view_n, 1):
            gt_ycbcr[0, i, j, :, :, :] = gt_image_ycbcr[i + num_vew_gap::view_ori, j + num_vew_gap::view_ori, :]

            for k in range(channel_n):
                img = Image.fromarray(gt_image_ycbcr[i + num_vew_gap::view_ori, j + num_vew_gap::view_ori, k])
                lr_ycbcr[0, i, j, :, :, k] = img.resize((image_w // scale, image_h // scale), Image.BICUBIC)

    return gt_ycbcr, lr_ycbcr


def data_prepare(dir_LF, view_n_ori, view_n_new, scale, disparity_list):
    assert view_n_new % 2 == 1
    D = len(disparity_list)
    gt_ycbcr, lr_ycbcr = image_input(dir_LF, scale, view_n_ori, view_n_new)
    gt_y = gt_ycbcr[..., 0]
    lr_y = lr_ycbcr[..., 0]
    _, U, V, lr_X, lr_Y = list(lr_y.shape)

    gt_y = gt_y.squeeze(0)

    lr_y = torch.from_numpy(lr_y.copy()).cuda().reshape(1, 1, -1, lr_X, lr_Y)
    time_strat = time.time()
    lr_y_sheared = warp_all(lr_y, disparity_list / scale, view_n_new, view_position=[view_n_new // 2, view_n_new // 2],
                            align_corners=False)
    lr_y_sheared = lr_y_sheared.reshape(D, U, V, lr_X, lr_Y).cpu().numpy()

    return lr_y_sheared / 255.0, gt_y / 255.0, lr_ycbcr, gt_ycbcr, time_strat


def data_prepare_real_LF(dir_LF, view_n_new, scale, disparity_list):
    assert view_n_new % 2 == 1
    D = len(disparity_list)

    LF_image = cv2.imread(dir_LF, cv2.IMREAD_UNCHANGED)[:, :, 0:3]
    LF_image = LF_image[:, :, ::-1]
    LF_image = color.rgb2ycbcr(LF_image).astype(np.float32)

    x, y, _ = LF_image.shape

    lr_ycbcr = np.zeros((1, view_n_new, view_n_new, x // view_n_new, y // view_n_new, 3), dtype=np.float32)
    for i in range(view_n_new):
        for j in range(view_n_new):
            lr_ycbcr[0, i, j] = LF_image[i::view_n_new, j::view_n_new]

    hr_image = cv2.imread(dir_LF[:-4] + '_hr.png', cv2.IMREAD_UNCHANGED)[:, :, 0:3]
    hr_image = hr_image[:, :, ::-1]
    hr_image = color.rgb2ycbcr(hr_image).astype(np.float32)

    lr_y = lr_ycbcr[..., 0]
    _, U, V, lr_X, lr_Y = list(lr_y.shape)

    lr_ycbcr_sheared = torch.from_numpy(lr_ycbcr.copy()).cuda().permute(0, 5, 1, 2, 3, 4).reshape(1, 3, -1, lr_X, lr_Y)
    time_strat = time.time()
    lr_ycbcr_sheared = warp_all(lr_ycbcr_sheared, disparity_list / scale, view_n_new,
                                view_position=[view_n_new // 2, view_n_new // 2],
                                align_corners=False)

    lr_ycbcr_sheared = lr_ycbcr_sheared.reshape(3, D, U, V, lr_X, lr_Y).cpu().numpy()

    return lr_ycbcr_sheared / 255.0, _, lr_ycbcr, hr_image / 255.0, time_strat
