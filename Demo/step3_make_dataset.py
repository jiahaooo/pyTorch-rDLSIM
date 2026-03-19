import os
import numpy as np
import torch

from utils.mrc import ReadMRC
from utils.option import mkdir, sir_parse2dict, save_json
from utils.image_process import F_cal_average_photon
from utils.tools import mkdir_with_time


def raw_weighted_np(x):
    if len(x.shape) == 7:
        (T, O, P, C, D, H, W) = x.shape
        for idx_T in range(T):
            for idx_C in range(C):
                for idx_D in range(D):
                    OPHW = torch.from_numpy(x[idx_T, :, :, idx_C, idx_D, :, :].copy())
                    OPHW = OPHW * torch.mean(OPHW[0, ...]) / torch.mean(OPHW, dim=[1, 2, 3], keepdim=True)
                    x[idx_T, :, :, idx_C, idx_D, :, :] = OPHW.numpy()
    else:
        raise NotImplementedError
    return x


def ParaProcess(npmat1, npmat2, sampling_rate):
    return np.concatenate((npmat1.flatten(), np.abs(npmat2).flatten(), np.angle(npmat2).flatten(), np.array([sampling_rate])), axis=0)


def ReadMrcImage(path):
    rm = ReadMRC(path)
    data = rm.get_total_data_as_mat(convert_to_tensor=False).astype(np.float32)
    return np.array(data)



def main():
    # ----------------------------------------
    #            <User Defined>
    # ----------------------------------------
    # # LifeAct
    # dataset_folder = r'K:\data_LifeAct_bioSR_copy'
    # min_allowed_photon = 50
    # max_allowed_photon = 65535  # raw
    # gt_raw_name = 'TIRFSIM488_GTRawData.mrc'
    # noisy_raw_name = 'TIRFSIM488_NoisyRawData.mrc'
    # gt_sim_name = 'TIRFSIM488_GTRawData_SIM.mrc'
    # noisy_sim_name = 'TIRFSIM488_NoisyRawData_SIM.mrc'
    # k0_name = 'TIRFSIM488_GTRawData_k0.npy'
    # phase_name = 'TIRFSIM488_GTRawData_phase.npy'
    # json_name = 'TIRFSIM488_GTRawData.json'

    # # Clathrin
    # dataset_folder = r'K:\data_Clathrin_bioSR_copy'
    # min_allowed_photon = 25
    # max_allowed_photon = 65535  # raw
    # gt_raw_name = 'TIRFSIM488_GTRawData.mrc'
    # noisy_raw_name = 'TIRFSIM488_NoisyRawData.mrc'
    # gt_sim_name = 'TIRFSIM488_GTRawData_SIM.mrc'
    # noisy_sim_name = 'TIRFSIM488_NoisyRawData_SIM.mrc'
    # k0_name = 'TIRFSIM488_GTRawData_k0.npy'
    # phase_name = 'TIRFSIM488_GTRawData_phase.npy'
    # json_name = 'TIRFSIM488_GTRawData.json'

    # # MAP7
    # dataset_folder = r'K:\data_MAP7_bioSR_copy'
    # min_allowed_photon = 25
    # max_allowed_photon = 65535  # raw
    # gt_raw_name = 'GISIM488_GTRawData.mrc'
    # noisy_raw_name = 'GISIM488_NoisyRawData.mrc'
    # gt_sim_name = 'GISIM488_GTRawData_SIM.mrc'
    # noisy_sim_name = 'GISIM488_NoisyRawData_SIM.mrc'
    # k0_name = 'GISIM488_GTRawData_k0.npy'
    # phase_name = 'GISIM488_GTRawData_phase.npy'
    # json_name = 'GISIM488_GTRawData.json'

    # KDEL
    dataset_folder = r'K:\data_KDEL_bioSR_copy'
    min_allowed_photon = 25
    max_allowed_photon = 65535  # raw
    gt_raw_name = 'GISIM488_GTRawData.mrc'
    noisy_raw_name = 'GISIM488_NoisyRawData.mrc'
    gt_sim_name = 'GISIM488_GTRawData_SIM.mrc'
    noisy_sim_name = 'GISIM488_NoisyRawData_SIM.mrc'
    k0_name = 'GISIM488_GTRawData_k0.npy'
    phase_name = 'GISIM488_GTRawData_phase.npy'
    json_name = 'GISIM488_GTRawData.json'


    # ----------------------------------------
    #            <mkdir and config-txt>
    # ----------------------------------------
    training_data_folder = mkdir_with_time(os.path.join(os.path.split(dataset_folder)[0], os.path.split(dataset_folder)[1] + '_DST'))

    for s_1 in ['train']:
        for s_2 in ['Raw_LSNR_1', 'SIM_LSNR_1', 'Raw_HSNR', 'SIM_HSNR', 'Para', 'Json']:
            mkdir(os.path.join(training_data_folder, s_1 + '_' + s_2))
    for s_1 in ['val']:
        for s_2 in ['Raw_LSNR_1', 'SIM_LSNR_1', 'Raw_HSNR', 'SIM_HSNR', 'Para', 'Json']:
            mkdir(os.path.join(training_data_folder, s_1 + '_' + s_2))

    # ----------------------------------------
    #            <Process and Save>
    # ----------------------------------------

    training_patch_num = 0
    val_patch_num = 0

    val_avg_photon_list = []
    for idx in range(10, 50):

        rm = ReadMRC(os.path.join(dataset_folder, '{:0>2}'.format(idx), gt_raw_name))
        json_dict = (sir_parse2dict(os.path.join(dataset_folder, '{:0>2}'.format(idx), json_name)))

        raw_hsnr = ReadMrcImage(os.path.join(dataset_folder, '{:0>2}'.format(idx), gt_raw_name)) - json_dict['camera_background']
        raw_lsnr_1 = ReadMrcImage(os.path.join(dataset_folder, '{:0>2}'.format(idx), noisy_raw_name)) - json_dict['camera_background']
        raw_hsnr, raw_lsnr_1 = raw_weighted_np(raw_hsnr), raw_weighted_np(raw_lsnr_1)
        sim_hsnr = ReadMrcImage(os.path.join(dataset_folder, '{:0>2}'.format(idx), gt_sim_name))
        sim_lsnr_1 = ReadMrcImage(os.path.join(dataset_folder, '{:0>2}'.format(idx), noisy_sim_name))

        para = ParaProcess(
            np.load(os.path.join(dataset_folder, '{:0>2}'.format(idx), k0_name)),
            np.load(os.path.join(dataset_folder, '{:0>2}'.format(idx), phase_name)),
            rm.opt['height_space_sampling']
        )

        for slice_num in range(raw_lsnr_1.shape[0]):

            average_photon = F_cal_average_photon(raw_lsnr_1[slice_num:slice_num + 1, ...], mean_axis=(0, 1, 2, 3, 4))
            if average_photon < min_allowed_photon or average_photon > max_allowed_photon:
                continue

            raw_lsnr_slice_1 = raw_lsnr_1[slice_num:slice_num + 1, ...].copy()
            raw_hsnr_slice = raw_hsnr[0:1, ...].copy()
            sim_lsnr_slice_1 = sim_lsnr_1[slice_num:slice_num + 1, ...].copy()
            sim_hsnr_slice = sim_hsnr[0:1, ...].copy()

            intensity_div_1 = np.mean(raw_hsnr_slice) / np.mean(raw_lsnr_slice_1)
            raw_hsnr_slice /= intensity_div_1
            sim_hsnr_slice /= intensity_div_1

            this_max = max(np.max(raw_hsnr_slice), np.max(raw_lsnr_slice_1))

            raw_lsnr_slice_1 /= this_max
            raw_hsnr_slice /= this_max
            sim_lsnr_slice_1 /= this_max
            sim_hsnr_slice /= this_max

            if 15 <= idx:
                # train
                np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), raw_lsnr_slice_1)
                np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), sim_lsnr_slice_1)
                np.save(os.path.join(training_data_folder, 'train_Raw_HSNR', '{:0>5}.npy'.format(training_patch_num)), raw_hsnr_slice)
                np.save(os.path.join(training_data_folder, 'train_SIM_HSNR', '{:0>5}.npy'.format(training_patch_num)), sim_hsnr_slice)
                np.save(os.path.join(training_data_folder, 'train_Para', '{:0>5}.npy'.format(training_patch_num)), para)
                save_json(json_dict, os.path.join(training_data_folder, 'train_Json', '{:0>5}.json'.format(training_patch_num)))
                training_patch_num += 1

            else:
                # val
                val_avg_photon_list.append(average_photon)
                np.save(os.path.join(training_data_folder, 'val_Raw_LSNR_1', '{:0>5}.npy'.format(val_patch_num)), raw_lsnr_slice_1)
                np.save(os.path.join(training_data_folder, 'val_SIM_LSNR_1', '{:0>5}.npy'.format(val_patch_num)), sim_lsnr_slice_1)
                np.save(os.path.join(training_data_folder, 'val_Raw_HSNR', '{:0>5}.npy'.format(val_patch_num)), raw_hsnr_slice)
                np.save(os.path.join(training_data_folder, 'val_SIM_HSNR', '{:0>5}.npy'.format(val_patch_num)), sim_hsnr_slice)
                np.save(os.path.join(training_data_folder, 'val_Para', '{:0>5}.npy'.format(val_patch_num)), para)
                save_json(json_dict, os.path.join(training_data_folder, 'val_Json', '{:0>5}.json'.format(val_patch_num)))
                val_patch_num += 1


if __name__ == '__main__':
    main()
