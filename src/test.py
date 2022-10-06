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
        '-d', '--datasets', type=str, default=None, dest='datasets',
        help='Loading 4D LF images (micro-lens and PNG format files) from this path: (default: %(default)s)')
    parser.add_argument(
        '-o', '--view_n_ori', type=int, default=14, dest='view_n_ori',
        help='Original data length: (default: %(default)s)')
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


def test_main(datasets, scale, view_n, view_n_ori, disparity_range, disparity_count, gpu_no=0, is_save=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    torch.backends.cudnn.benchmark = True

    print('=' * 40)
    print('build network...')

    num_layers = 6
    channels = 32
    u_net_channel = 16
    if scale == 4:
        up_num_layers = 6
        u_net_num_layers = 3
    else:
        up_num_layers = 4
        u_net_num_layers = 4

    model = LFHSR_mask(scale=scale, an=view_n, mask_num_layers=num_layers, mask_channel=channels,
                       fusion_num_layers=num_layers, fusion_channel=channels,
                       u_net_num_layers=u_net_num_layers, u_net_channel=u_net_channel, up_num_layers=up_num_layers,
                       up_channel=channels)

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

    xls_list = []
    psnr_list = []
    ssim_list = []
    time_list = []

    cut_margin = 16

    image_list = []
    for line in open(f'../data_list/{datasets}.txt'):
        image_list.append(line.strip())
    image_list.sort()

    for index, image_name in enumerate(image_list):
        print('-' * 100)
        print('[{}/{}]'.format(index + 1, len(image_list)), image_name)

        lr_y_sheared, gt_y, lr_ycbcr, gt_ycbcr, time_strat = utils_test.data_prepare(
            f'../LFHSR_Datasets/Test/{image_name}.png',
            view_n_ori, view_n, scale,
            disparity_list)

        hr_img = gt_y[view_n // 2, view_n // 2]
        hr_y = predict_y(lr_y_sheared, hr_img, model, view_n, scale, disparity_list)
        hr_y = hr_y.reshape(view_n, view_n, hr_y.shape[-2], hr_y.shape[-1])

        time_ = time.time() - time_strat

        time_list.append(time_)

        hr_y = hr_y.cpu().numpy()
        hr_y = np.clip(hr_y, 16.0 / 255.0, 235.0 / 255.0)

        gt_y = np.clip(gt_y, 16.0 / 255.0, 235.0 / 255.0)

        psnr_view_list = []
        ssim_view_list = []

        for i in range(view_n):
            for j in range(view_n):
                if i == view_n // 2 and j == view_n // 2:
                    print('      -/-      ', end='\t\t')
                    continue
                if cut_margin == 0:
                    psnr_view = peak_signal_noise_ratio(hr_y[i, j, :, :],
                                                        gt_y[i, j, :, :], data_range=1)
                    psnr_view_list.append(psnr_view)
                    ssim_view = structural_similarity(hr_y[i, j, :, :],
                                                      gt_y[i, j, :, :], data_range=1)
                    ssim_view_list.append(ssim_view)
                else:
                    psnr_view = peak_signal_noise_ratio(hr_y[i, j, cut_margin:-cut_margin, cut_margin:-cut_margin],
                                                        gt_y[i, j, cut_margin:-cut_margin,
                                                        cut_margin:-cut_margin], data_range=1)
                    psnr_view_list.append(psnr_view)
                    ssim_view = structural_similarity(hr_y[i, j, cut_margin:-cut_margin, cut_margin:-cut_margin],
                                                      gt_y[i, j, cut_margin:-cut_margin, cut_margin:-cut_margin],
                                                      data_range=1)
                    ssim_view_list.append(ssim_view)
                print('{:6.4f}/{:6.4f}'.format(psnr_view, ssim_view), end='\t\t')
            print('')

        if is_save:
            result_image_path = save_path + image_name + '/'
            if not os.path.exists(result_image_path):
                os.makedirs(result_image_path)

            hr_cbcr = gt_ycbcr[..., 1:3]
            for i in range(view_n):
                for j in range(view_n):
                    hr_y_item = np.clip(hr_y[i, j, :, :] * 255.0, 16.0, 235.0)
                    hr_y_item = hr_y_item[:, :, np.newaxis]

                    hr_ycbcr_item = np.concatenate((hr_y_item, hr_cbcr[0, i, j]), 2)
                    hr_rgb_item = color.ycbcr2rgb(hr_ycbcr_item) * 255.0

                    hr_rgb_item = hr_rgb_item[:, :, ::-1]
                    img_save_path = result_image_path + str(i) + str(j) + '.png'
                    cv2.imwrite(img_save_path, hr_rgb_item)

        psnr_ = np.mean(psnr_view_list)
        psnr_list.append(psnr_)
        ssim_ = np.mean(ssim_view_list)
        ssim_list.append(ssim_)

        print('PSNR: {:.4f} SSIM: {:.4f} time: {:.4f}'.format(psnr_, ssim_, time_))
        xls_list.append([image_name, psnr_, ssim_, time_])

    xls_list.append(['average', np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)])
    xls_list = np.array(xls_list)

    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'time'])
    result.to_csv(
        save_path + 'result_s{}n{}_{}_{}_{}.csv'.format(scale, view_n, datasets, disparity_range, int(time.time())))

    print('-' * 100)
    print('Average: PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f}'.format(np.mean(psnr_list), np.mean(ssim_list),
                                                                     np.mean(time_list)))
    print('all done')


def predict_y(lr_y_shear, hr_img, model, view_n, scale, disparity_list):
    with torch.no_grad():
        lr_y_shear = torch.from_numpy(lr_y_shear.copy()).cuda()
        hr_img = torch.from_numpy(hr_img.copy()).cuda()
        hr_img = hr_img.unsqueeze(0).unsqueeze(0)

        lr_y_shear = lr_y_shear.reshape(1, -1, view_n * view_n, lr_y_shear.shape[3], lr_y_shear.shape[4])

        if lr_y_shear.shape[3] * scale > 300:
            hr_y = utils_test.overlap_crop_forward(lr_y_shear, hr_img, scale, disparity_list, model, max_length=200,
                                                   shave=10, mod=8)

        else:
            lr_LF = lr_y_shear[:, len(disparity_list) // 2].reshape(1, view_n, view_n, lr_y_shear.shape[3],
                                                                    lr_y_shear.shape[4])
            hr_y = model(lr_y_shear, lr_LF, hr_img, disparity_list)

        return hr_y


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    test_main(
        datasets=args.datasets,
        scale=args.scale,
        view_n=args.view_n,
        view_n_ori=args.view_n_ori,
        disparity_range=args.disparity_range,
        disparity_count=args.disparity_count,
        gpu_no=args.gpu_no,
        is_save=args.is_save
    )
