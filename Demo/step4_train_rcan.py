import os
import sys

sys.path.append(os.path.split(os.getcwd())[0])

from rDL_logic.main_train_plain import main
from utils.option import AI_parse_read, AI_parse_process

if __name__ == '__main__':
    PATH_LIST = [
        # r'K:\data_LifeAct_bioSR_copy_DST',
        # r'K:\data_Clathrin_bioSR_copy_DST',
        # r'K:\data_MAP7_bioSR_copy_DST',
        r'K:\data_KDEL_bioSR_copy_DST',
    ]

    for path in PATH_LIST:
        opt = AI_parse_read('../rDL_options/train_recon2d_rcan.json')
        opt['dataset_root'] = path
        opt = AI_parse_process(opt)
        main(json_path=None, opt=opt)
