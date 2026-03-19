import torch.nn as nn
import torch
from rDL_model.base_rcan import RCAN_Group, CALayer
import time
# -------------------------------------------------------------------------
#       rDL (Qiao, 2021)
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
# Math:
#
# (0) SIM Imaging Model:
#
#                 y = Ax
#
# (1) Train a super-resolution network (marked as f_1) mapping raw data (y) into GT (x)
#
#                 f_1 = argmin_f || y - f(x) ||_1
#
# (2) Get predicted SR images and use the imaging model:
#
#                 x_{pred} = f_1(y)
#    
#                 y_{simu} = Ax_{pred}
# 
# (3) Two-stream Raw Denoise (per three channel)
#
#                 y_{pred} = f_4(f_2(y), f_3(y_{simu}))
#
# < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < <
# -------------------------------------------------------------------------


class RDL_RCAN_Denoiser(nn.Module):
    # paras: 5.186755 M
    # GFLOPs: 4041.128146944
    # [4090] fps 3.4770744408599885 Hz
    # [4090] time 287.5980992091354 ms
    def __init__(self, in_nc, out_nc, para1=5, para2=5):
        n_feats = 64
        n_group= para1
        n_block = para2
        super(RDL_RCAN_Denoiser, self).__init__()
        self.FeatureExtract_REAL = RCAN_Raw2Feature(in_nc=in_nc, n_feats=n_feats, n_resgroups=n_group, n_resblocks=n_block)
        self.FeatureExtract_SIMU = RCAN_Raw2Feature(in_nc=in_nc, n_feats=n_feats, n_resgroups=n_group//2, n_resblocks=n_block)
        self.Reconstruction = RCAN_Feature2result(n_feats=n_feats, out_nc=out_nc, n_resgroups=n_group, n_resblocks=n_block)

    def forward(self, raw_real, raw_simu):
        fm_real = self.FeatureExtract_REAL(raw_real)
        fm_simu = self.FeatureExtract_SIMU(raw_simu)
        raw = self.Reconstruction(fm_real + fm_simu)
        return raw


class RCAN_Raw2Feature(nn.Module):
    def __init__(self, in_nc, n_feats, n_resgroups, n_resblocks):
        super(RCAN_Raw2Feature, self).__init__()

        modules_head = [nn.Conv2d(in_nc, n_feats, (3,3), (1,1), (1,1), bias=True)]
        self.head = nn.Sequential(*modules_head)

        modules_body = [nn.LeakyReLU(0.2)]
        for _ in range(n_resgroups):
            modules_body.append(RCAN_Group(n_feats, reduction=16, act=nn.LeakyReLU(0.2), n_resblocks=n_resblocks))
        self.body = nn.Sequential(*modules_body)

        modules_tail = [nn.Conv2d(n_feats, n_feats, (3,3), (1,1), (1,1), bias=True), nn.LeakyReLU(0.2)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        return x

class RCAN_Feature2result(nn.Module):
    def __init__(self, out_nc, n_feats, n_resgroups, n_resblocks):
        super(RCAN_Feature2result, self).__init__()

        modules_head = [nn.Conv2d(n_feats, n_feats, (3,3), (1,1), (1,1), bias=True), nn.LeakyReLU(0.2)]
        self.head = nn.Sequential(*modules_head)

        modules_body = []
        for _ in range(n_resgroups):
            modules_body.append(RCAN_Group(n_feats, reduction=16, act=nn.LeakyReLU(0.2), n_resblocks=n_resblocks))
        self.body = nn.Sequential(*modules_body)

        modules_tail = [nn.Conv2d(n_feats, n_feats * 4, (3,3), (1,1), (1,1), bias=True),
                        nn.LeakyReLU(0.2),
                        CALayer(4*n_feats, reduction=16),
                        nn.Conv2d(n_feats * 4, out_nc, (3,3), (1,1), (1,1), bias=True),
                        nn.LeakyReLU(0.2)
                        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        return x