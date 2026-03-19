import torch.nn as nn
import torch
import time
from rDL_model.base_rcan import RCAN_Group

class RCAN(nn.Module):
    # 4*4 - 72 layers
    # paras: 1.529153 M
    # GFlops: 398.442112
    # [4090] fps 39.45871411848944 Hz
    # [4090] time 25.34294445067644 ms
    #
    # 10*10 - 414 layers
    # paras: 8.003345 M
    # GFlops: 2081.331142912
    # [4090] fps 7.432254666901394 Hz
    # [4090] time 134.54867261927575 ms
    #
    # 10*20 - 814 layers
    # paras: 15.446945 M
    # GFlops: 4015.744205312
    # [4090] fps 3.8096129025077263 Hz
    # [4090] time 262.4938610801473 ms
    #
    # num of layer = n_resgroups * (4 * n_resblocks + 1) + 4
    def __init__(self, in_nc, out_nc, para1=None, para2=None, scale=2, each_ori=False, each_pha=False):
        super(RCAN, self).__init__()

        if each_ori or each_pha:  assert scale == 1
        assert (not each_ori) or (not each_pha)

        self.scale = scale
        self.num_phase = in_nc
        self.each_ori = each_ori
        self.each_pha = each_pha

        n_resgroups = para1 if para1 is not None else 4
        n_resblocks = para2 if para2 is not None else 4
        n_feats = 64

        # define head module
        modules_head = [nn.Conv2d(in_nc, n_feats, (3,3), (1,1), (1,1), bias=True)]
        self.head = nn.Sequential(*modules_head)

        # define body module
        modules_body = []
        for _ in range(n_resgroups):
            modules_body.append(RCAN_Group(n_feats, reduction=16, act=nn.ReLU(True), n_resblocks=n_resblocks))
        modules_body.append(nn.Conv2d(n_feats, n_feats, (3,3), (1,1), (1,1), bias=True))
        self.body = nn.Sequential(*modules_body)

        # define tail module
        if scale == 1:
            modules_tail = [nn.Conv2d(n_feats, n_feats, (3,3), (1,1), (1,1), bias=True), nn.Conv2d(n_feats, out_nc, (3,3), (1,1), (1,1), bias=True)]
        elif scale == 2:
            modules_tail = [nn.Conv2d(n_feats, n_feats * 4, (3,3), (1,1), (1,1), bias=True), nn.PixelShuffle(2), nn.Conv2d(n_feats, out_nc, (3,3), (1,1), (1,1), bias=True)]
        else: raise NotImplementedError
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, *args, **kwargs):

        [B, C, H, W] = x.shape

        if self.each_pha:
            x = x.reshape(B * C, 1, H, W)
        elif self.each_ori:
            x = x.reshape(B*(C//self.num_phase), self.num_phase, H, W)

        x = self.head(x)
        res = self.body(x)
        res += x

        res = self.tail(res)

        if self.each_ori: res = res.reshape(B, C, H, W)

        return res