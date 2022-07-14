import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class encode_base(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(encode_base, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=scale,
                              padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class decode_base(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(decode_base, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=scale + 2,
                                       stride=scale, padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class U_Net_reference(nn.Module):
    def __init__(self, num_layer, in_channels, channel):
        super(U_Net_reference, self).__init__()
        encode_part = []
        for i in range(num_layer):
            if i == 0:
                encode_part.append(
                    encode_base(in_channels=in_channels + 1, out_channels=2 ** (i + 1) * channel, scale=2))
            else:
                encode_part.append(
                    encode_base(in_channels=2 ** i * channel + 1, out_channels=2 ** (i + 1) * channel, scale=2))
        self.encode_part = nn.ModuleList(encode_part)

        decode_part = []
        for i in range(num_layer):
            if i == 0:
                decode_part.append(decode_base(in_channels=2 ** num_layer * channel,
                                               out_channels=2 ** (num_layer - 1) * channel, scale=2))
            else:
                decode_part.append(decode_base(in_channels=2 ** (num_layer - i + 1) * channel,
                                               out_channels=2 ** (num_layer - i - 1) * channel, scale=2))
        self.decode_part = nn.ModuleList(decode_part)

        self.mask = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.num_layer = num_layer

    def forward(self, hr_LF, hr_img):
        encode_item = []
        hr_img_tmp = hr_img
        for i in range(self.num_layer):
            hr_LF = self.encode_part[i](torch.cat((hr_LF, hr_img_tmp), dim=1))
            encode_item.append(hr_LF)
            hr_img_tmp = F.interpolate(hr_img, scale_factor=1 / (2 ** (i + 1)), mode='bicubic',
                                       recompute_scale_factor=True,
                                       align_corners=False)
        for i in range(self.num_layer):
            if i == 0:
                hr_LF = self.decode_part[i](hr_LF)
            else:
                hr_LF = self.decode_part[i](torch.cat((hr_LF, encode_item[-i - 1]), dim=1))

        return self.mask(hr_LF)


class upSample(nn.Module):
    def __init__(self, scale, is_cnn=True):
        super(upSample, self).__init__()
        if is_cnn:
            self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=(scale + 2, scale + 2), stride=(scale, scale),
                                               padding=(1, 1), output_padding=(0, 0), bias=True)
        self.is_cnn = is_cnn
        self.scale = scale

    def forward(self, input):
        h, w = list(input.shape[-2:])
        input_shape = list(input.shape[:-2])
        input_shape.append(h * self.scale)
        input_shape.append(w * self.scale)
        input = input.reshape(-1, 1, h, w)
        if self.is_cnn:
            return self.upsample(input).reshape(*input_shape)
        else:
            return F.interpolate(input, scale_factor=self.scale,
                                 mode='bicubic', recompute_scale_factor=True, align_corners=False).reshape(*input_shape)


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(res_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        return self.PReLU(x + self.conv(x))


class mask_part(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels=1, scale=None, channel=32):
        super(mask_part, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channel, kernel_size=3, padding=1, stride=1))
        layers.append(nn.PReLU())
        for i in range(num_layers - 1):
            layers.append(res_block(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, stride=1))

        if scale is None:
            layers.append(
                nn.Conv2d(in_channels=channel, out_channels=out_channels, kernel_size=3, padding=1, stride=1))
        else:
            layers.append(nn.ConvTranspose2d(in_channels=channel, out_channels=out_channels, kernel_size=scale + 2,
                                             stride=scale, padding=1))

        self.mask_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.mask_conv(x)


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)

    return torch.from_numpy(filter).float()


class AltFilter(nn.Module):
    def __init__(self, an, channel=32):
        super(AltFilter, self).__init__()

        self.an = an
        self.relu0 = nn.PReLU()
        self.relu1 = nn.PReLU()
        self.spaconv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.relu0(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.reshape(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * h * w, c, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.relu1(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.reshape(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * self.an * self.an, c, h, w)  # [N*an2,c,h,w]
        return out


class net4x(nn.Module):

    def __init__(self, an, layer, channel=32):

        super(net4x, self).__init__()

        self.an = an
        self.an2 = an * an
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=3, stride=1, padding=1)

        self.altblock1 = self.make_layer(layer_num=layer, channel=channel)
        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.altblock2 = self.make_layer(layer_num=layer, channel=channel)
        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, layer_num, channel):
        layers = []
        for i in range(layer_num):
            layers.append(AltFilter(self.an, channel=channel))
        return nn.Sequential(*layers)

    def forward(self, lr):

        N, _, _, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.relu(self.conv0(lr))  # [N*81,64,h,w]
        f_1 = self.altblock1(x)  # [N*81,64,h,w]
        fup_1 = self.fup1(f_1)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]

        f_2 = self.altblock2(fup_1)  # [N*81,64,2h,2w]
        fup_2 = self.fup2(f_2)  # [N*81,64,4h,4w]
        res_2 = self.res2(fup_2)  # [N*81,1,4h,4w]
        iup_2 = self.iup2(sr_2x)  # [N*81,1,4h,4w]
        sr_4x = res_2 + iup_2  # [N*81,1,4h,4w]

        sr_4x = sr_4x.view(N, self.an2, h * 4, w * 4)

        return sr_4x


class net8x(nn.Module):

    def __init__(self, an, layer, channel=32):

        super(net8x, self).__init__()

        self.an = an
        self.an2 = an * an
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=3, stride=1, padding=1)

        self.altblock1 = self.make_layer(layer_num=layer, channel=channel)
        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.altblock2 = self.make_layer(layer_num=layer, channel=channel)
        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.altblock3 = self.make_layer(layer_num=layer, channel=channel)
        self.fup3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res3 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, layer_num, channel):
        layers = []
        for i in range(layer_num):
            layers.append(AltFilter(self.an, channel=channel))
        return nn.Sequential(*layers)

    def forward(self, lr):

        N, _, _, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.relu(self.conv0(lr))  # [N*81,64,h,w]
        f_1 = self.altblock1(x)  # [N*81,64,h,w]
        fup_1 = self.fup1(f_1)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]

        f_2 = self.altblock2(fup_1)  # [N*81,64,2h,2w]
        fup_2 = self.fup2(f_2)  # [N*81,64,4h,4w]
        res_2 = self.res2(fup_2)  # [N*81,1,4h,4w]
        iup_2 = self.iup2(sr_2x)  # [N*81,1,4h,4w]
        sr_4x = res_2 + iup_2  # [N*81,1,4h,4w]

        f_3 = self.altblock3(fup_2)  # [N*81,64,2h,2w]
        fup_3 = self.fup3(f_3)  # [N*81,64,4h,4w]
        res_3 = self.res3(fup_3)  # [N*81,1,4h,4w]
        iup_3 = self.iup3(sr_4x)  # [N*81,1,4h,4w]
        sr_8x = res_3 + iup_3  # [N*81,1,4h,4w]

        sr_8x = sr_8x.view(N, self.an2, h * 8, w * 8)

        return sr_8x


class LFHSR_mask(nn.Module):
    def __init__(self, scale, an, mask_num_layers=6, mask_channel=32, fusion_num_layers=6, fusion_channel=32,
                 u_net_num_layers=3, u_net_channel=16, up_num_layers=5, up_channel=32):
        super(LFHSR_mask, self).__init__()
        in_channels = an * an
        self.mask_part = mask_part(num_layers=mask_num_layers, in_channels=in_channels, scale=scale,
                                   channel=mask_channel)

        self.mask_refine_part = U_Net_reference(num_layer=u_net_num_layers, in_channels=1,
                                                channel=u_net_channel)
        self.fusion_part = mask_part(num_layers=fusion_num_layers, in_channels=2, channel=fusion_channel)

        if scale == 4:
            self.upsample = net4x(an=an, layer=up_num_layers, channel=up_channel)
        elif scale == 8:
            self.upsample = net8x(an=an, layer=up_num_layers, channel=up_channel)

        self.scale = scale
        self.an = an

        assert self.an % 2 == 1

    def warp_reverse_all(self, img, disparity_list, view_n):
        B, C, D, X, Y = list(img.shape)
        UV = view_n * view_n
        img = img.permute(2, 0, 1, 3, 4).reshape(D, B * C, X, Y)  # D ,BC, X, Y
        view_central = view_n // 2
        ref_position = np.array([view_central, view_central])

        img_all = []
        for i in range(view_n):
            for j in range(view_n):
                theta = []
                for disparity in disparity_list:
                    target_position = np.array([i, j])
                    d = (ref_position - target_position) * disparity * 2
                    theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]])
                    theta.append(theta_t.unsqueeze(0))
                theta = torch.cat(theta, 0).cuda()
                grid = F.affine_grid(theta, img.size(), align_corners=False)
                img_tmp = F.grid_sample(img, grid, align_corners=False)
                img_all.append(img_tmp.unsqueeze(0))
        img_all = torch.cat(img_all, dim=0)
        img_all = img_all.reshape(UV, D, B, C, X, Y).permute(2, 3, 1, 0, 4, 5)  # B, C, D, UV, X, Y
        return img_all

    def forward(self, lr_LF_shear, lr_LF, hr_img, disparity_list):
        B, D, UV, X, Y = list(lr_LF_shear.shape)

        hr_LF = self.upsample(lr_LF).reshape(B, UV, self.scale * X, self.scale * Y)

        lr_LF_shear = lr_LF_shear.reshape(B * D, UV, X, Y)  # BD, UV, X, Y

        hr_mask = self.mask_part(lr_LF_shear - lr_LF_shear[:, UV // 2:UV // 2 + 1])

        hr_mask = hr_mask.reshape(B, D, self.scale * X, self.scale * Y)
        hr_mask = torch.softmax(hr_mask, dim=1)
        hr_mask = hr_mask.reshape(B * D, 1, self.scale * X, self.scale * Y)

        hr_img = hr_img.expand(-1, D, -1, -1).reshape(B * D, 1, self.scale * X, self.scale * Y)
        hr_mask = self.mask_refine_part(hr_mask, hr_img)
        hr_mask = hr_mask.reshape(B, D, 1, self.scale * X, self.scale * Y)
        hr_mask = torch.softmax(hr_mask, dim=1)
        hr_mask = hr_mask.permute(0, 2, 1, 3, 4)

        hr_mask_deshear = self.warp_reverse_all(hr_mask, -disparity_list, self.an)

        hr_img = hr_img.reshape(B, 1, D, self.scale * X, self.scale * Y)
        hr_img_deshear = self.warp_reverse_all(hr_img, -disparity_list, self.an)

        hr_fusion_LF = None
        for i in range(len(disparity_list)):
            if i == 0:
                hr_fusion_LF = hr_img_deshear[:, :, i] * hr_mask_deshear[:, :, i]
            else:
                hr_fusion_LF = hr_fusion_LF * (1 - hr_mask_deshear[:, :, i]) + hr_img_deshear[:, :,
                                                                               i] * hr_mask_deshear[:, :, i]

        hr_fusion_LF = hr_fusion_LF.reshape(-1, 1, self.scale * X, self.scale * Y)
        hr_LF = hr_LF.reshape(-1, 1, self.scale * X, self.scale * Y)

        hr_LF = self.fusion_part(torch.cat((hr_fusion_LF, hr_LF), dim=1))

        if self.training:
            hr_mask_view = torch.sigmoid(torch.sum(hr_mask_deshear, dim=2)).reshape(-1, 1, self.scale * X,
                                                                                    self.scale * Y)

            return hr_LF.reshape(B, -1, self.scale * X, self.scale * Y), \
                   hr_mask_view.reshape(B, -1, self.scale * X, self.scale * Y)
        else:
            return hr_LF.reshape(B, -1, self.scale * X, self.scale * Y)
