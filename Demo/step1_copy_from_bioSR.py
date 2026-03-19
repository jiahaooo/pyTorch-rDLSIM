import torch
import os

from utils.mrc import ReadMRC, WriteMRC
from utils.tools import mkdir


def main1():

    # MAP7
    biosr_path = r'E:\Common\BioSR-DataSet\Microtubules_SIRecon'
    output_path = r'K:\data_MAP7_bioSR_copy'
    output_hsnr_name = 'GISIM488_GTRawData.mrc'
    output_lsnr_name = 'GISIM488_NoisyRawData.mrc'

    mkdir(output_path)
    rangelist = []
    for i in range(1, 9 + 1):
        rangelist.append(i)

    for idx in range(1, 50 + 1):
        rm_hsnr = ReadMRC(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawSIMData_gt.mrc'))
        header = rm_hsnr.header
        big_endian = rm_hsnr.big_endian

        mkdir(os.path.join(output_path, '{:0>2d}'.format(idx)))

        # gt
        raw_hsnr = rm_hsnr.get_total_data_as_mat()

        wm_hsnr = WriteMRC(os.path.join(output_path, '{:0>2d}'.format(idx), output_hsnr_name), header, big_endian)
        wm_hsnr.write_data_append(raw_hsnr)

        # noisy
        raw_lsnr_list = []
        for idx_level in rangelist:
            print(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawSIMData_level_{:0>2d}.mrc'.format(idx_level)))
            rm_lsnr = ReadMRC(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawSIMData_level_{:0>2d}.mrc'.format(idx_level)))
            raw_lsnr_list.append(rm_lsnr.get_total_data_as_mat())

        raw_lsnr = torch.stack(raw_lsnr_list).squeeze(1)
        wm_lsnr = WriteMRC(os.path.join(output_path, '{:0>2d}'.format(idx), output_lsnr_name), header, big_endian)
        wm_lsnr.write_data_append(raw_lsnr)


def main2():
    # KDEL
    biosr_path = r'E:\Common\BioSR-DataSet\ER_SIRecon'
    output_path = r'K:\data_KDEL_bioSR_copy'
    output_hsnr_name = 'GISIM488_GTRawData.mrc'
    output_lsnr_name = 'GISIM488_NoisyRawData.mrc'

    mkdir(output_path)
    rangelist = []
    for i in range(1, 6 + 1):
        rangelist.append(i)

    for idx in range(1, 50 + 1):

        rm_hsnr = ReadMRC(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawGTSIMData', 'RawGTSIMData_level_01.mrc'))
        header = rm_hsnr.header
        big_endian = rm_hsnr.big_endian

        gt_raw_lsnr_list = []
        for idx_level in rangelist:
            print(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawSIMData', 'RawGTSIMData_level_{:0>2d}.mrc'.format(idx_level)))
            mkdir(os.path.join(output_path, '{:0>2d}'.format(idx)))
            rm_hsnr = ReadMRC(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawGTSIMData', 'RawGTSIMData_level_{:0>2d}.mrc'.format(idx_level)))
            gt_raw_lsnr_list.append(rm_hsnr.get_total_data_as_mat())
        raw_hsnr = torch.stack(gt_raw_lsnr_list).squeeze(1)
        wm_hsnr = WriteMRC(os.path.join(output_path, '{:0>2d}'.format(idx), output_hsnr_name), header, big_endian)
        wm_hsnr.write_data_append(raw_hsnr)

        noisy_raw_lsnr_list = []
        for idx_level in rangelist:
            rm_lsnr = ReadMRC(os.path.join(biosr_path, 'Cell_{:0>3d}'.format(idx), 'RawSIMData', 'RawSIMData_level_{:0>2d}.mrc'.format(idx_level)))
            noisy_raw_lsnr_list.append(rm_lsnr.get_total_data_as_mat())
        raw_lsnr = torch.stack(noisy_raw_lsnr_list).squeeze(1)
        wm_lsnr = WriteMRC(os.path.join(output_path, '{:0>2d}'.format(idx), output_lsnr_name), header, big_endian)
        wm_lsnr.write_data_append(raw_lsnr)


if __name__ == '__main__':
    # main1()
    main2()
