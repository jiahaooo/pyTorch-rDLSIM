import random
from math import pi

import torch
from torch.optim import lr_scheduler, Adam, AdamW
import torch.nn.functional as F

from utils.option import sir_parse2dict, dict2nonedict
from utils.tools import my_meshgrid

from rDL_model.common import generate_pattern
from rDL_model.select_network import define_G, define_P
from rDL_model.model_base import ModelBase
from rDL_model.loss import GP_loss_fun

device = torch.device('cuda')


class ModelRDL(ModelBase):
    # ------------------------------------
    # define network and optimizer
    # ------------------------------------
    def __init__(self, opt):
        super(ModelRDL, self).__init__(opt)

        self.netG = define_G(opt).to(self.device)
        self.netP = define_P(opt).to(self.device)

        self.G_lossfn = None
        self.P_lossfn = None
        self.G_optimizer = None
        self.P_optimizer = None

        self.need_blur = False if opt['need_blur'] is False else True

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.netP.train()
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = {}  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['pretrained_netG']
        if load_path_G is not None:
            self.opt['outinfo'].info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_P = self.opt['pretrained_netP']
        if load_path_P is not None:
            self.opt['outinfo'].info('Loading model for P [{:s}] ...'.format(load_path_P))
            self.load_network(load_path_P, self.netP)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label, type_label):
        if type_label == 'G':
            self.save_network(self.save_dir, self.netG, 'G', iter_label)
        elif type_label == 'P':
            self.save_network(self.save_dir, self.netP, 'P', iter_label)
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define G_loss and P_loss
    # ----------------------------------------
    def define_loss(self):
        self.G_lossfn = GP_loss_fun(self.opt['G_lossfn_type'])
        self.P_lossfn = GP_loss_fun(self.opt['P_lossfn_type'])

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):

        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                self.opt['outinfo'].info('Params [{:s}] will not optimize.'.format(k))
        weight_decay = self.opt['G_optimizer_weight_decay'] if self.opt['G_optimizer_weight_decay'] is not None else 0.
        if self.opt['G_optimizer_type'].lower() == 'adam':
            assert weight_decay == 0, "Incorrect implementation of Weight Decay in ADAM optimizer"
            self.G_optimizer = Adam(G_optim_params, lr=self.opt['G_optimizer_lr'], weight_decay=weight_decay)
        elif self.opt['G_optimizer_type'].lower() == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt['G_optimizer_lr'], weight_decay=weight_decay)
        else:
            raise RuntimeError

        P_optim_params = []
        for k, v in self.netP.named_parameters():
            if v.requires_grad:
                P_optim_params.append(v)
            else:
                self.opt['outinfo'].info('Params [{:s}] will not optimize.'.format(k))
        weight_decay = self.opt['P_optimizer_weight_decay'] if self.opt['P_optimizer_weight_decay'] is not None else 0.
        if self.opt['P_optimizer_type'].lower() == 'adam':
            assert weight_decay == 0, "Incorrect implementation of Weight Decay in ADAM optimizer"
            self.P_optimizer = Adam(P_optim_params, lr=self.opt['P_optimizer_lr'], weight_decay=weight_decay)
        elif self.opt['P_optimizer_type'].lower() == 'adamw':
            self.P_optimizer = AdamW(P_optim_params, lr=self.opt['P_optimizer_lr'], weight_decay=weight_decay)
        else:
            raise RuntimeError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers['G'] = lr_scheduler.MultiStepLR(self.G_optimizer, self.opt['G_scheduler_milestones'], self.opt['G_scheduler_gamma'])
            self.schedulers['P'] = lr_scheduler.MultiStepLR(self.P_optimizer, self.opt['P_scheduler_milestones'], self.opt['P_scheduler_gamma'])
        elif self.opt['G_scheduler_type'] == 'CosineAnnealingLR':
            self.schedulers['G'] = lr_scheduler.CosineAnnealingLR(self.G_optimizer, T_max=self.opt['G_scheduler_IterNum'], eta_min=self.opt['G_scheduler_MinLR'])
            self.schedulers['P'] = lr_scheduler.CosineAnnealingLR(self.P_optimizer, T_max=self.opt['P_scheduler_IterNum'], eta_min=self.opt['P_scheduler_MinLR'])
        else:
            raise RuntimeError

    # ----------------------------------------
    # feed in/out data
    # ----------------------------------------
    def feed_data_train(self, data):
        opt = self.opt
        if opt['need_para']: self.para_data = data['para'].to(self.device)
        self.json_path = data['json_path']

        self.raw_input_data = data['train_RAW_LSNR_1'].to(self.device)
        if opt['supervise'] in ["full-supervised"]:
            self.raw_target_data = data['train_RAW_HSNR'].to(self.device)
            self.sim_target_data = data['train_SIM_HSNR'].to(self.device)
        else:
            raise NotImplementedError

    def feed_data_val(self, data):
        opt = self.opt
        if opt['need_para']: self.para_data = data['para'].to(self.device)
        self.json_path = data['json_path']
        self.raw_input_data = data['val_RAW_LSNR_1'].to(self.device)
        if opt['supervise'] in ["full-supervised"]:
            self.raw_target_data = data['val_RAW_HSNR'].to(self.device)
            self.sim_target_data = data['val_SIM_HSNR'].to(self.device)
        else:
            raise NotImplementedError

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if current_step < self.opt['P_pretrain_iternum']:
            # ------------------------------------
            # only optimize P
            # ------------------------------------
            self.netP.train()
            self.P_optimizer.zero_grad()
            self.sim_infer_data = self.netP(self.raw_input_data, self.para_data, self.json_path)
            P_loss = self.P_lossfn(self.sim_infer_data, self.sim_target_data)
            P_loss.backward()
            self.P_optimizer.step()
            self.update_learning_rate('P')  # 1-P
            self.log_dict['P_loss'] = P_loss.item()
        else:
            # self.cuda_memory_allocated = None
            # ------------------------------------
            # only optimize G
            # ------------------------------------
            with torch.no_grad():
                self.netP.eval()
                pred_sim = self.netP(self.raw_input_data, self.para_data, self.json_path)
            simu_raw = self.add_pattern_rdl(pred_sim.detach(), 'train', self.need_blur)

            self.G_optimizer.zero_grad()
            n_ori, n_pha = self.opt['raw_shape_OPC'][0], self.opt['raw_shape_OPC'][1]
            batchsize_old = self.raw_input_data.shape[0]

            if self.opt['rdl_train_dataType'] == 'perOri':

                if self.opt['rdl_train_subBatch'] >= n_ori:

                    if len(self.raw_input_data.shape) == 4:  # 2d
                        shape_after_reshape = (batchsize_old * n_ori, n_pha, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])

                    else:
                        raise NotImplementedError

                    self.raw_input_data = self.raw_input_data.reshape(*shape_after_reshape)
                    simu_raw = simu_raw.reshape(*shape_after_reshape)
                    self.raw_target_data = self.raw_target_data.reshape(*shape_after_reshape)

                else:

                    if len(self.raw_input_data.shape) == 4:  # 2d
                        shape_after_reshape = (batchsize_old, n_ori, n_pha, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                        shape_after_choice = (batchsize_old * self.opt['rdl_train_subBatch'], n_pha, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])

                    else:
                        raise NotImplementedError

                    choice_list = list(range(n_ori * 1))
                    random.shuffle(choice_list)
                    choice_list = choice_list[:self.opt['rdl_train_subBatch']]

                    self.raw_input_data = (self.raw_input_data.reshape(*shape_after_reshape)[:, choice_list]).reshape(shape_after_choice)
                    simu_raw = (simu_raw.reshape(*shape_after_reshape)[:, choice_list]).reshape(shape_after_choice)
                    self.raw_target_data = (self.raw_target_data.reshape(*shape_after_reshape)[:, choice_list]).reshape(shape_after_choice)

            elif self.opt['rdl_train_dataType'] == 'perPha':

                if self.opt['rdl_train_subBatch'] >= n_ori * n_pha:

                    if len(self.raw_input_data.shape) == 4:
                        shape_after_reshape = (batchsize_old * n_ori * n_pha, 1, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    elif len(self.raw_input_data.shape) == 5:
                        shape_after_reshape = (batchsize_old * n_ori * n_pha, 1, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    else:
                        raise NotImplementedError

                    self.raw_input_data = self.raw_input_data.reshape(*shape_after_reshape)
                    simu_raw = simu_raw.reshape(*shape_after_reshape)
                    self.raw_target_data = self.raw_target_data.reshape(*shape_after_reshape)

                else:

                    if len(self.raw_input_data.shape) == 4:
                        shape_after_reshape = (batchsize_old, n_ori * n_pha, 1, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                        shape_after_choice = (batchsize_old * self.opt['rdl_train_subBatch'], 1, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    elif len(self.raw_input_data.shape) == 5:
                        shape_after_reshape = (batchsize_old, n_ori * n_pha, 1, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                        shape_after_choice = (
                        batchsize_old * self.opt['rdl_train_subBatch'], 1, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    else:
                        raise NotImplementedError

                    choice_list = list(range(n_ori * n_pha))
                    random.shuffle(choice_list)
                    choice_list = choice_list[:self.opt['rdl_train_subBatch']]

                    self.raw_input_data = (self.raw_input_data.reshape(*shape_after_reshape)[:, choice_list]).reshape(shape_after_choice)
                    simu_raw = (simu_raw.reshape(*shape_after_reshape)[:, choice_list]).reshape(shape_after_choice)
                    self.raw_target_data = (self.raw_target_data.reshape(*shape_after_reshape)[:, choice_list]).reshape(shape_after_choice)

            else:
                raise NotImplementedError

            self.raw_infer_data = self.netG(self.raw_input_data, simu_raw)

            G_loss = self.G_lossfn(self.raw_infer_data, self.raw_target_data)
            G_loss.backward()
            self.G_optimizer.step()
            self.update_learning_rate('G')  # 0-G
            self.log_dict['G_loss'] = G_loss.item()

    # ----------------------------------------
    # add pattern
    # ----------------------------------------
    def add_pattern_rdl(self, imSIM, state, need_blur=True):
        if self.opt['data'] == '2d-sim':
            result_list = []
            for idx_bs in range(imSIM.shape[0]):
                para_data = self.para_data[idx_bs]
                assert len(para_data) == 6 + 3 + 3 + 1
                opt_json_path = self.json_path[idx_bs]
                Nx, Ny = self.raw_input_data[idx_bs].shape[-1], self.raw_input_data[idx_bs].shape[-2]
                cur_k0 = para_data[0:6].reshape(3, 2)
                phase_list = - para_data[9:12]
                cur_k0_angle = torch.atan2(cur_k0[:, 1], cur_k0[:, 0])
                cur_k0 = torch.sqrt(torch.sum(torch.square(cur_k0), 1))
                opt = sir_parse2dict(opt_json_path)
                ndirs, nphases = opt['num_orientation'], opt['num_phase']
                phase_space = 2 * pi / nphases
                xx = opt['width_space_sampling'] * torch.arange(-Nx // 2, Nx // 2, 1, device=device)
                yy = opt['height_space_sampling'] * torch.arange(-Ny // 2, Ny // 2, 1, device=device)
                [Y, X] = my_meshgrid(yy, xx)
                if need_blur:
                    OTF = self.set_otf_while_training(opt, Nx, Ny, 1, state)
                gen_raw_list = []
                imSIM_slice = F.interpolate(imSIM[idx_bs:idx_bs + 1], (Ny, Nx)).squeeze()
                for idx_d in range(ndirs):
                    alpha = cur_k0_angle[idx_d]
                    for idx_p in range(nphases):
                        kxL = cur_k0[idx_d] * pi * torch.cos(alpha)
                        kyL = cur_k0[idx_d] * pi * torch.sin(alpha)
                        kxR = -cur_k0[idx_d] * pi * torch.cos(alpha)
                        kyR = -cur_k0[idx_d] * pi * torch.sin(alpha)
                        phOffset = phase_list[idx_d] + idx_p * phase_space
                        interBeam = torch.exp(1j * (kxL * X + kyL * Y + phOffset)) + torch.exp(1j * (kxR * X + kyR * Y))
                        pattern = torch.square(torch.abs(interBeam)) / 4.0
                        if need_blur:
                            temp = torch.fft.fftshift(torch.fft.fft2(pattern * imSIM_slice)) * OTF
                            Generated_img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(temp)))
                        else:
                            Generated_img = pattern * imSIM_slice
                        gen_raw_list.append(Generated_img)
                result_list.append(torch.stack(gen_raw_list))  # [C, H, W]
            return torch.stack(result_list)

        else:
            raise NotImplementedError

    # # ----------------------------------------
    # # val / inference
    # # ----------------------------------------
    def val(self):
        self.netG.eval()
        self.netP.eval()
        with torch.no_grad():

            pred_sim = self.netP(self.raw_input_data, self.para_data, self.json_path)

            simu_raw = self.add_pattern_rdl(pred_sim, 'test', self.need_blur)

            n_ori, n_pha = self.opt['raw_shape_OPC'][0], self.opt['raw_shape_OPC'][1]

            if self.opt['rdl_train_dataType'] == 'perOri':
                if len(self.raw_input_data.shape) == 4:
                    shape_after_reshape = (n_ori, n_pha, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    shape_before_reshape = (1, n_ori * n_pha, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                elif len(self.raw_input_data.shape) == 5:
                    shape_after_reshape = (n_ori, n_pha, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    shape_before_reshape = (1, n_ori * n_pha, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                else:
                    raise NotImplementedError
            elif self.opt['rdl_train_dataType'] == 'perPha':
                if len(self.raw_input_data.shape) == 4:
                    shape_after_reshape = (n_ori * n_pha, 1, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    shape_before_reshape = (1, n_ori * n_pha, self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                elif len(self.raw_input_data.shape) == 5:
                    shape_after_reshape = (n_ori * n_pha, 1, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                    shape_before_reshape = (1, n_ori * n_pha, self.raw_input_data.shape[-3], self.raw_input_data.shape[-2], self.raw_input_data.shape[-1])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            # self.raw_infer_data = self.netG(self.raw_input_data.reshape(*shape_after_reshape), simu_raw.reshape(*shape_after_reshape)).reshape(*shape_before_reshape)

            temp1, temp2 = self.raw_input_data.reshape(*shape_after_reshape), simu_raw.reshape(*shape_after_reshape)
            temp = []
            for idx in range(temp1.shape[0]):
                temp.append(self.netG(temp1[idx:idx + 1], temp2[idx:idx + 1]))
            self.raw_infer_data = torch.stack(temp).reshape(*shape_before_reshape)

        self.netG.train()
        self.netP.train()

    # ----------------------------------------
    # get batch results (first slice in batch):
    # ----------------------------------------
    def current_visuals(self, need_input=True, need_target=True):
        out_dict = dict2nonedict({})
        opt = self.opt
        # raw_shape = (1, *opt['raw_shape_OPC'])
        # sim_shape = (1, *opt['sim_shape_OPC'])
        raw_shape = opt['raw_shape_OPC']
        sim_shape = opt['sim_shape_OPC']
        json_class = sir_parse2dict(self.json_path[0])
        out_dict['raw_sampling_rate'] = [json_class['width_space_sampling'], json_class['height_space_sampling'], json_class['depth_space_sampling']]
        out_dict['sim_sampling_rate'] = [json_class['width_space_sampling'] / opt['sim_scale'], json_class['height_space_sampling'] / opt['sim_scale'],
                                         json_class['depth_space_sampling']]

        if need_input:
            out_dict['sim_input_data'] = self.tensor2format(self.conv_rec(self.raw_input_data.detach()[0], self.para_data.detach()[0], raw_shape, json_class), sim_shape)
            out_dict['raw_input_data'] = self.tensor2format(self.raw_input_data.detach()[0], raw_shape)
        if need_target:
            out_dict['sim_target_data'] = self.tensor2format(self.sim_target_data.detach()[0], sim_shape)
            # out_dict['sim_target_data'] = self.tensor2format(self.conv_rec(self.raw_target_data.detach()[0], self.para_data.detach()[0], raw_shape, json_class), sim_shape)
            out_dict['raw_target_data'] = self.tensor2format(self.raw_target_data.detach()[0], raw_shape)
        out_dict['sim_infer_data'] = self.tensor2format(self.conv_rec(self.raw_infer_data.detach()[0], self.para_data.detach()[0], raw_shape, json_class), sim_shape)
        out_dict['raw_infer_data'] = self.tensor2format(self.raw_infer_data.detach()[0], raw_shape)

        return out_dict
