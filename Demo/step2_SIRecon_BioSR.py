import torch
import os

from utils.option import dict2str, sir_parse2dict, save_json
from utils.tools import WriteOutputTxt, timeblock
from utils.mrc import ReadMRC, WriteMRC, make_sr_header

from SIR_core.base import make_otf_2d
from SIR_core.pe import guess_k0, force_modamp, save_wave_vector, SIMEstimate2D
from SIR_core.r2 import SIMReconstr2D

device = torch.device('cuda')


def main():
    # # LifeAct
    # path = r'K:\data_LifeAct_bioSR_copy'
    # user_defined_longexp_name = ['TIRFSIM488_GTRawData.mrc']
    # user_defined_shortexp_name = ['TIRFSIM488_NoisyRawData.mrc']
    # json_file = 'biosr_tirf_488.json'

    # Clathrin
    path = r'K:\data_Clathrin_bioSR_copy'
    user_defined_longexp_name = ['TIRFSIM488_GTRawData.mrc']
    user_defined_shortexp_name = ['TIRFSIM488_NoisyRawData.mrc']
    json_file = 'biosr_tirf_488.json'

    # # MAP7
    # path = r'K:\data_MAP7_bioSR_copy'
    # user_defined_longexp_name = ['GISIM488_GTRawData.mrc']
    # user_defined_shortexp_name = ['GISIM488_NoisyRawData.mrc']
    # json_file = 'biosr_lownagi_488.json'

    # # KDEL
    # path = r'K:\data_KDEL_bioSR_copy'
    # user_defined_longexp_name = ['GISIM488_GTRawData.mrc']
    # user_defined_shortexp_name = ['GISIM488_NoisyRawData.mrc']
    # json_file = 'biosr_lownagi_488.json'

    found_longexp_files = []
    found_shortexp_files = []
    num_avg_in_para_esti = 1

    # ----------------------------------------
    #            <Pre Processing>
    # ----------------------------------------
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            for idx in range(len(user_defined_longexp_name)):
                if os.path.exists(os.path.join(path, folder, user_defined_longexp_name[idx])) and \
                        os.path.exists(os.path.join(path, folder, user_defined_shortexp_name[idx])):
                    found_longexp_files.append(os.path.join(path, folder, user_defined_longexp_name[idx]))
                    found_shortexp_files.append(os.path.join(path, folder, user_defined_shortexp_name[idx]))

    # ----------------------------------------
    #            <ITER Processing>
    # ----------------------------------------
    for idx in range(len(found_longexp_files)):
        with timeblock('reconstruct:  ' + found_longexp_files[idx]):

            mrc_file = found_longexp_files[idx]
            opt_json = sir_parse2dict(os.path.join('../SIR_options', json_file))  # return a class
            raw = ReadMRC(mrc_file, opt=opt_json)  # <opt append>
            opt = raw.opt
            save_json(opt, mrc_file[:-4] + '.json')
            opt['outinfo'] = WriteOutputTxt(mrc_file[:-4] + '.txt')
            opt['outinfo'].info('\n-------- all parameters --------\n')
            opt['outinfo'].info(dict2str(opt) + '\n')
            k0_guess = guess_k0(opt, device=device)
            otf = make_otf_2d(opt, k0=k0_guess, device=device)

            opt['outinfo'].info('\n-------- do estimate pattern --------\n')
            opt['outinfo'].info('load one tps data\n')
            data = raw.get_timepoint_data_as_mat(begin_timepoint=0, read_timepoint=num_avg_in_para_esti).mean(axis=0, keepdims=True)
            sem = SIMEstimate2D(k0_guess=k0_guess, opt=opt, otf=otf, device=device)

            with torch.no_grad():
                wave_vector = sem.esti(data)
            if opt['if_force_modamp']: wave_vector = force_modamp(wave_vector, opt)
            save_wave_vector(wave_vector=wave_vector, path=mrc_file)

            srm = SIMReconstr2D(opt=opt, wave_vector=wave_vector, otf=otf, device=device)

            opt['outinfo'].info('\n-------- do reconstruction --------\n')
            CW = WriteMRC(mrc_file[:-4] + r'_SIM.mrc', make_sr_header(raw.header, opt))
            batch_idx = 0
            while True:
                data = raw.get_next_timepoint_batch(batchsize=1)
                if data is None:
                    break
                else:
                    pass
                opt['outinfo'].info('load batch, index: {:d}, size: {:d} and do reconstruction\n'.format(batch_idx, data.shape[0]))
                with torch.no_grad():
                    result = srm.reconstr(data)  # [TOPCDHW] -> [TDHW]
                CW.write_data_append(result.unsqueeze(1).unsqueeze(1).unsqueeze(1))  # [TDHW] -> [TOPCDHW]
                batch_idx += 1

            # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
            mrc_file = found_shortexp_files[idx]
            save_wave_vector(wave_vector=wave_vector, path=mrc_file)
            opt_json = sir_parse2dict(os.path.join('../SIR_options', json_file))  # return a class
            raw = ReadMRC(mrc_file, opt=opt_json)  # <opt append>
            opt = raw.opt
            save_json(opt, mrc_file[:-4] + '.json')
            opt['outinfo'] = WriteOutputTxt(mrc_file[:-4] + '.txt')
            opt['outinfo'].info('\n-------- all parameters --------\n')
            opt['outinfo'].info(dict2str(opt) + '\n')
            opt['outinfo'].info('\n-------- do estimate pattern --------\n')
            opt['outinfo'].info('pattern parameters are given by ones estimated with high-snr data \n')
            opt['outinfo'].info('\n-------- do reconstruction --------\n')
            CW = WriteMRC(mrc_file[:-4] + r'_SIM.mrc', make_sr_header(raw.header, opt))

            batch_idx = 0
            while True:
                data = raw.get_next_timepoint_batch(batchsize=1)
                if data is None:
                    break
                else:
                    pass
                opt['outinfo'].info('load batch, index: {:d}, size: {:d} and do reconstruction\n'.format(batch_idx, data.shape[0]))
                with torch.no_grad():
                    result = srm.reconstr(data)  # [TOPCDHW] -> [TDHW]
                CW.write_data_append(result.unsqueeze(1).unsqueeze(1).unsqueeze(1))  # [TDHW] -> [TOPCDHW]
                batch_idx += 1
        # break


if __name__ == '__main__':
    main()
