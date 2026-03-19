import numpy as np
import platform
import argparse
import torch
import random
import math
import os
import time
import sys
import threading

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from torch.utils.data import DataLoader

sys.path.append(os.path.split(os.getcwd())[0])

from rDL_logic.main_train_rdl import main
from utils.option import AI_parse, AI_parse_read, AI_parse_process

if __name__ == '__main__':

    opt = AI_parse_read('../RDL_options/train_rdl2d.json')
    # opt['dataset_root'] = r'K:\data_KDEL_bioSR_copy_DST'
    # opt['dataset_root'] = r'K:\data_Clathrin_bioSR_copy_DST'
    opt['dataset_root'] = r'K:\data_MAP7_bioSR_copy_DST'
    # opt['dataset_root'] = r'K:\data_LifeAct_bioSR_copy_DST'
    opt['pretrained_netP'] = os.path.join(opt['dataset_root'], 'full-supervised_reconstruction_rcan', 'model', 'PSNR_BEST_G.pth')
    opt['G_scheduler_IterNum'] = 200000
    opt = AI_parse_process(opt)
    main(json_path=None, opt=opt)
