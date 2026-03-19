import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp

device = torch.device('cuda')


class GP_loss_fun(torch.nn.Module):
    def __init__(self, lossfn_type):
        super(GP_loss_fun, self).__init__()
        lossfn_type = lossfn_type.lower()
        while lossfn_type.find(' ') >= 0:
            lossfn_type = lossfn_type.replace(' ', '')
        lossfn_type = lossfn_type.split('+')
        self.loss = []
        self.weight = []
        for this_loss in lossfn_type:
            if this_loss.find('*') >= 0:
                self.weight.append(float(this_loss[:this_loss.find('*')]))
                this_loss = this_loss[this_loss.find('*') + 1:].lower()
            else:
                self.weight.append(1.0)
                this_loss = this_loss.lower()

            if False:
                pass

            elif this_loss in ['mse', 'l2']:
                self.loss.append(nn.MSELoss().to(device))
            elif this_loss in ['mae', 'l1']:
                self.loss.append(nn.L1Loss().to(device))
            elif this_loss == 'ssim':
                self.loss.append(SSIMLoss().to(device))
            else:
                raise NotImplementedError('Loss type not found')

    def forward(self, x, y):
        # x = infer, y = target
        loss = 0
        for idx in range(len(self.loss)):
            loss += self.weight[idx] * self.loss[idx](x, y)
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.data_range = data_range
        self.channel = 1
        window = self.create_window(window_size, sigma, self.channel)
        self.register_buffer('window', window)

    @staticmethod
    def create_window(window_size, sigma, channel):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        _1d = gauss.unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _get_window(self, x):
        c = x.size(1)
        if c != self.channel or self.window.dtype != x.dtype or self.window.device != x.device:
            window = self.create_window(self.window_size, self.sigma, c).to(device=x.device, dtype=x.dtype)
            self.window = window
            self.channel = c
        return self.window

    def _filter(self, x, window):
        p = self.window_size // 2
        x = F.pad(x, (p, p, p, p), mode='reflect')
        return F.conv2d(x, window, groups=x.size(1))

    def forward(self, img1, img2):
        window = self._get_window(img1)

        mu1 = self._filter(img1, window)
        mu2 = self._filter(img2, window)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self._filter(img1 * img1, window) - mu1_sq
        sigma2_sq = self._filter(img2 * img2, window) - mu2_sq
        sigma12 = self._filter(img1 * img2, window) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            ssim = ssim_map.mean()
        else:
            ssim = ssim_map.mean(dim=(1, 2, 3))

        return 1.0 - ssim

