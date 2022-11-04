import torch
import os
import time
import numpy as np
import pandas as pd
import cv2
import skimage.color as color
import sys
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from argparse import ArgumentParser, ArgumentTypeError

import utils_test
from model import LFHSR_mask


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Unsupported value encountered.')


def opts_parser():
    usage = "LF-Hybrid-SR"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-i', '--image_path', type=str, default=None, dest='image_path',
        help='Loading Hybrid-lens LF images from this path: (default: %(default)s)')
    parser.add_argument(
        '-s', '--save_path', type=str, default='../result/', dest='save_path',
        help='Save upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-S', '--scale', type=int, default=4, dest='scale',
        help='up-sampling scale factor : (default: %(default)s)')
    parser.add_argument(
        '-N', '--view_n', type=int, default=9, dest='view_n',
        help='Angular resolution of input LFs for test: (default: %(default)s)')
    parser.add_argument(
        '-r', '--disparity_range', type=int, default=4, dest='disparity_range',
        help='disparity range of the LF images, if disparity range is [-4, 3], please set range=4: (default: %(default)s)')
    parser.add_argument(
        '-c', '--disparity_count', type=int, default=32, dest='disparity_count',
        help='the number of the planes in MPIs: (default: %(default)s)')
    parser.add_argument(
        '-g', '--gpu_no', type=int, default=0, dest='gpu_no',
        help='GPU used: (default: %(default)s)')
    parser.add_argument(
        '-sv', '--is_save', type=str2bool, default=False, dest='is_save',
        help='Save the results of LF images: (default: %(default)s)')

    return parser


def test_main(image_path, scale, view_n, disparity_range, disparity_count, gpu_no=0, is_save=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    torch.backends.cudnn.benchmark = True

    print('=' * 40)
    print('build network...')

    num_layers = 6
    channels = 32
    u_net_channel = 16
    if scale == 2:
        up_num_layers = 6
        u_net_num_layers = 3
    elif scale == 4:
        up_num_layers = 6
        u_net_num_layers = 3
    else:
        up_num_layers = 4
        u_net_num_layers = 4

    model = LFHSR_mask(scale=scale, an=view_n, mask_num_layers=num_layers, mask_channel=channels,
                       fusion_num_layers=num_layers, fusion_channel=channels, u_net_num_layers=u_net_num_layers,
                       u_net_channel=u_net_channel, up_num_layers=up_num_layers, up_channel=channels)

    utils_test.get_parameter_number(model)

    model.cuda()
    model.eval()
    print('done')

    print('=' * 40)
    print('load model...')

    state_dict = torch.load(f'../pretrain_model/LFHSR_{scale}_{view_n}.pkl')
    model.load_state_dict(state_dict)
    print('done')

    print('=' * 40)
    print('create save directory...')
    save_path = f'../result/s{scale}n{view_n}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sys.stdout = Logger(save_path + 'test_{}.log'.format(int(time.time())), sys.stdout)
    print('path: ', save_path)
    print('done')

    print('=' * 40)
    print('predict image...')

    disparity_min = -disparity_range
    disparity_max = disparity_range
    disparity_count = (disparity_max - disparity_min) / disparity_count
    disparity_list = np.arange(disparity_min, disparity_max + disparity_count, disparity_count)

    time_list = []

    image_list_all = os.listdir(image_path)
    image_list = []
    for image_name in image_list_all:
        if image_name[-6:] == 'hr.png':
            continue
        image_list.append(image_name)
    image_list.sort()

    for index, image_name in enumerate(image_list):
        print('-' * 100)
        print('[{}/{}]'.format(index + 1, len(image_list)), image_name)

        lr_ycbcr_sheared, _, lr_ycbcr, hr_img, time_strat = utils_test.data_prepare_real_LF(
            image_path + image_name, view_n, scale, disparity_list)

        hr_y = predict_y(lr_ycbcr_sheared[0:1], hr_img[..., 0], model, view_n, scale, disparity_list)
        hr_y = hr_y.reshape(view_n, view_n, hr_y.shape[-2], hr_y.shape[-1])

        time_ = time.time() - time_strat
        time_list.append(time_)

        hr_y = hr_y.cpu().numpy()
        hr_y = np.clip(hr_y, 16.0 / 255.0, 235.0 / 255.0)

        result_image_path = save_path + image_name[0:-4] + '/'
        if not os.path.exists(result_image_path):
            os.makedirs(result_image_path)

        hr_cbcr = predict_cbcr(lr_ycbcr[:, :, :, :, :, 1:3], scale, view_n)
        for i in range(view_n):
            for j in range(view_n):
                hr_y_item = np.clip(hr_y[i, j, :, :] * 255.0, 16.0, 235.0)
                hr_y_item = hr_y_item[:, :, np.newaxis]

                hr_ycbcr_item = np.concatenate((hr_y_item, hr_cbcr[0, i, j]), 2)
                hr_rgb_item = color.ycbcr2rgb(hr_ycbcr_item) * 255.0

                hr_rgb_item = hr_rgb_item[:, :, ::-1]
                img_save_path = result_image_path + str(i) + str(j) + '.png'
                cv2.imwrite(img_save_path, hr_rgb_item)

    print('all done')


def predict_y(lr_y_shear, hr_img, model, view_n, scale, disparity_list):
    with torch.no_grad():
        lr_y_shear = torch.from_numpy(lr_y_shear.copy()).cuda()
        hr_img = torch.from_numpy(hr_img.copy()).cuda()
        hr_img = hr_img.unsqueeze(0).unsqueeze(0)

        lr_y_shear = lr_y_shear.reshape(1, -1, view_n * view_n, lr_y_shear.shape[-2], lr_y_shear.shape[-1])

        if lr_y_shear.shape[3] * scale > 300:
            hr_y = utils_test.overlap_crop_forward(lr_y_shear, hr_img, scale, disparity_list, model, max_length=500,
                                                   shave=10, mod=8)

        else:
            lr_LF = lr_y_shear[:, len(disparity_list) // 2].reshape(1, view_n, view_n, lr_y_shear.shape[3],
                                                                    lr_y_shear.shape[4])
            hr_y = model(lr_y_shear, lr_LF, hr_img, disparity_list)

        return hr_y


def predict_cbcr(lr_cbcr, scale, view_n):
    hr_cbcr = np.zeros((1, view_n, view_n, lr_cbcr.shape[3] * scale, lr_cbcr.shape[4] * scale, 2))

    for i in range(view_n):
        for j in range(view_n):
            image_bicubic = cv2.resize(lr_cbcr[0, i, j, :, :, :],
                                       (lr_cbcr.shape[4] * scale, lr_cbcr.shape[3] * scale),
                                       interpolation=cv2.INTER_CUBIC)
            hr_cbcr[0, i, j, :, :, :] = image_bicubic
    return hr_cbcr


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    test_main(
        image_path=args.image_path,
        scale=args.scale,
        view_n=args.view_n,
        disparity_range=args.disparity_range,
        disparity_count=args.disparity_count,
        gpu_no=args.gpu_no,
        is_save=args.is_save
    )
