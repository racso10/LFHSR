import torch
import time
import os
import numpy as np
import utils_train
import sys
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentTypeError

from initializers import weights_init_xavier
from model import LFHSR_mask
from datasets import LFHSR_shear_Dataset


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
        '-S', '--scale', type=int, default=4, dest='scale',
        help='up-sampling scale factor : (default: %(default)s)')
    parser.add_argument(
        '-N', '--view_n', type=int, default=9, dest='view_n',
        help='Angular resolution of input LFs for training: (default: %(default)s)')
    parser.add_argument(
        '-dmin', '--disparity_min', type=int, default=-4, dest='disparity_min',
        help='the minimum disparity of the LF images: (default: %(default)s)')
    parser.add_argument(
        '-dmax', '--disparity_max', type=int, default=4, dest='disparity_max',
        help='the maximum disparity of the LF images: (default: %(default)s)')
    parser.add_argument(
        '-dg', '--disparity_grad', type=float, default=0.25, dest='disparity_grad',
        help='the disparity grad of the MPIs: (default: %(default)s)')
    parser.add_argument(
        '-l1', '--mask_num_layers', type=int, default=6, dest='mask_num_layers',
        help='the number of the layers in mask network: (default: %(default)s)')
    parser.add_argument(
        '-l2', '--fusion_num_layers', type=int, default=6, dest='fusion_num_layers',
        help='the number of the layers in mask network: (default: %(default)s)')
    parser.add_argument(
        '-l3', '--up_num_layers', type=int, default=6, dest='up_num_layers',
        help='the number of the layers in up-sample network: (default: %(default)s)')
    parser.add_argument(
        '-l4', '--u_net_num_layers', type=int, default=6, dest='u_net_num_layers',
        help='the number of the layers in u-net network: (default: %(default)s)')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=2, dest='batch_size')
    parser.add_argument(
        '-crop', '--crop_size', type=int, default=32, dest='crop_size',
        help='the crop_size of the training: (default: %(default)s)')
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=0.001, dest='base_lr')
    parser.add_argument(
        '-step', '--step_size', type=int, default=1600, dest='step_size',
        help='Learning rate decay every n epochs : (default: %(default)s)')
    parser.add_argument(
        '-g', '--gpu_no', type=int, default=0, dest='gpu_no',
        help='GPU used: (default: %(default)s)')

    return parser


MAX_EPOCH = 10000


def main(view_n, scale, disparity_min, disparity_max, disparity_grad, mask_num_layers, fusion_num_layers, up_num_layers,
         u_net_num_layers, batch_size, crop_size, base_lr, step_size, gpu_no):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

    torch.backends.cudnn.benchmark = True

    # dir_LFimages = './LFHSR_Datasets/Train/'
    dir_LFimages = '../../Dataset/LFHSR_hci_train_npy_99/'
    dir_save_name = '../net_store/'

    ''' Define Model(set parameters)'''

    channels = 32
    u_net_channel = 16

    model = LFHSR_mask(scale=scale, an=view_n, mask_num_layers=mask_num_layers, mask_channel=channels,
                       fusion_num_layers=fusion_num_layers, fusion_channel=channels,
                       u_net_num_layers=u_net_num_layers, u_net_channel=u_net_channel, up_num_layers=up_num_layers,
                       up_channel=channels)

    model.apply(weights_init_xavier)
    utils_train.get_parameter_number(model)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    if not os.path.exists(dir_save_name):
        os.makedirs(dir_save_name)
    else:
        print('Exist!')

    test_gap = 10

    disparity_list = np.arange(disparity_min, disparity_max + disparity_grad, disparity_grad)

    train_dataset = LFHSR_shear_Dataset(dir_LFimages, view_n_input=view_n, scale=scale,
                                        disparity_list=disparity_list, crop_size=crop_size, if_flip=True,
                                        if_rotation=True, is_all=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    current_iter = 0
    for epoch in range(current_iter, MAX_EPOCH):

        ''' Validation during the training process'''
        if epoch % test_gap == 0:
            torch.save(model.state_dict(), dir_save_name + 'LFHSR_{}_{}.pkl'.format(scale, str(view_n)))

        ''' Training begin'''
        current_iter, train_loss = train_shear(train_loader, model, epoch, view_n, disparity_list, optimizer, scheduler,
                                               current_iter)


def train_shear(train_loader, model, epoch, view_n, disparity_list, optimizer, scheduler, current_iter):
    model.train()

    time_start = time.time()
    total_loss = 0

    count = 0
    for lr_LF_shear, lr_LF, hr_img, view_position, hr_gt, hr_gt_mask in tqdm(train_loader):
        lr_LF_shear, lr_LF, hr_img, hr_gt, hr_gt_mask = lr_LF_shear.cuda(), lr_LF.cuda(), hr_img.cuda(), hr_gt.cuda(), hr_gt_mask.cuda()
        lr_LF_shear = lr_LF_shear.reshape(lr_LF_shear.shape[0], -1, view_n * view_n, lr_LF_shear.shape[-2],
                                          lr_LF_shear.shape[-1])
        hr_img = hr_img.unsqueeze(1)
        hr_gt = hr_gt.reshape(hr_gt.shape[0], -1, hr_gt.shape[-2], hr_gt.shape[-1])
        hr_pred, hr_mask_view = model(lr_LF_shear, lr_LF, hr_img, disparity_list)

        loss = utils_train.weight_L1Loss(hr_pred, hr_gt, 1 + 0.001 * (1 - hr_mask_view))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

    current_iter += 1
    scheduler.step()

    time_end = time.time()
    print('=========================================================================')
    print('Train Epoch: {} Learning rate: {:.2e} Time: {:.2f}s Average Loss: {:.6f} '
          .format(epoch, scheduler.get_last_lr()[0], time_end - time_start, total_loss / count))
    return current_iter, total_loss / count


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    main(
        scale=args.scale,
        view_n=args.view_n,
        disparity_min=args.disparity_min,
        disparity_max=args.disparity_max,
        disparity_grad=args.disparity_grad,
        mask_num_layers=args.mask_num_layers,
        fusion_num_layers=args.fusion_num_layers,
        up_num_layers=args.up_num_layers,
        u_net_num_layers=args.u_net_num_layers,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        base_lr=args.base_lr,
        step_size=args.step_size,
        gpu_no=args.gpu_no,
    )

    # -S 4 -N 9 -dmin 4 -dmax 4 -dg 0.25 -l1 6 -l2 6 -l3 6 -l4 3 -b 4 -crop 32 -lr 0.001 -step 1600 -g 0
    # -S 8 -N 9 -dmin 4 -dmax 4 -dg 0.25 -l1 6 -l2 6 -l3 4 -l4 4 -b 2 -crop 24 -lr 0.0005 -step 1600 -g 0
