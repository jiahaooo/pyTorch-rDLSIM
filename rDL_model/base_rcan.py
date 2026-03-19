import torch
import math
from math import pi
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda')


# ----------------------------------------
# Residual Channel Attention Group, Block, Layer
# ----------------------------------------
class CALayer(nn.Module):  # 2
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, (1, 1), padding=(0, 0), bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CALayer_3D(nn.Module):  # 2
    def __init__(self, channel, reduction):
        super(CALayer_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_du = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.ReLU(True),
            nn.Conv3d(channel // reduction, channel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):  # 4
    def __init__(self, n_feat, reduction, bias, act):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, (3, 3), (1, 1), (1, 1), bias=bias), act,
            nn.Conv2d(n_feat, n_feat, (3, 3), (1, 1), (1, 1), bias=bias), CALayer(n_feat, reduction))

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



class RCAN_Group(nn.Module):  # 4 * n_resblocks + 1
    def __init__(self, n_feat, reduction, act, n_resblocks):
        super(RCAN_Group, self).__init__()
        modules_body = []
        for _ in range(n_resblocks):
            modules_body.append(RCAB(n_feat, reduction, bias=True, act=act))
        modules_body.append(nn.Conv2d(n_feat, n_feat, (3, 3), (1, 1), (1, 1), bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
