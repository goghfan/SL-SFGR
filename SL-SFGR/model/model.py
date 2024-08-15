import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from . import metrics as Metrics
from model.BaseModel.Models.Encoder import Encoder

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))

        self.schedule_phase = None
        self.centered = opt['datasets']['centered']

        # set loss and load resume state
        self.set_loss()
        # self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.log_dict = OrderedDict()
            self.load_network()
        else:
            self.load_network()
        self.print_network(self.netG)

    def feed_data(self, data):
        self.data = self.set_device(data)
    def optimize_parameters_origin(self):
        self.optG.zero_grad()
        #让netG返回加噪后的数据，同时输入M F，输出的时候注意通道对应和连接问题
        loss = self.netG(self.data)
        #CT loss
        #l_ngf,l_ncc,l_reg,l_dice_label,l_dice_vessel,l_seg_m,l_seg_f = loss
        #l_mse,l_reg,l_dice,l_seg_m,l_seg_f = loss
        #l_tot = (l_ngf.mean() + l_ncc.mean() + 0.1*l_reg.mean() +l_dice_label.mean()+l_dice_vessel.mean()+ l_seg_m.mean() +l_seg_f.mean())
        #l_tot = (l_mse.mean()*10 + 0.1*l_reg.mean() +l_dice.mean()+ l_seg_m.mean() +l_seg_f.mean())

        #MRI loss
        #l_ncc,l_seg,l_dice,l_reg  = loss
        # l_ncc,l_seg,l_dice,l_reg  = loss
        l_tot = loss.mean()
        l_tot.backward()
        self.optG.step()
        # 记录各个损失项到日志
        # CT Loss
        # self.log_dict['l_mse'] = 10 * l_mse.mean().item()
        # self.log_dict['l_ngf'] = l_ngf.mean().item()
        # self.log_dict['l_ncc'] = l_ncc.mean().item()
        # self.log_dict['l_reg'] = 0.1 * l_reg.mean().item()
        # self.log_dict['l_dice_label'] = l_dice_label.mean().item()
        # self.log_dict['l_dice_vessel'] = l_dice_vessel.mean().item()
        # self.log_dict['l_seg_m'] = l_seg_m.mean().item()
        # self.log_dict['l_seg_f'] = l_seg_f.mean().item()
        # MRI Loss
        # self.log_dict['l_ncc'] = l_ncc.mean().item()
        # self.log_dict['l_ncc'] = l_ncc.mean().item()
        # self.log_dict['l_seg'] = 0.2*l_seg.mean().item()
        # self.log_dict['l_dice'] = l_dice.mean().item()
        # self.log_dict['l_reg'] = l_reg.mean().item()
        # 记录总的损失到日志
        self.log_dict['l_tot'] = l_tot.item()         
    def optimize_parameters(self):
        self.optG.zero_grad()
        #让netG返回加噪后的数据，同时输入M F，输出的时候注意通道对应和连接问题
        loss = self.netG(self.data)

        #dice版本
        l_ncc,l_seg,l_dice,l_reg  = loss
        l_tot = (l_ncc.mean() +0.2* l_seg.mean() +l_dice.mean()+ l_reg.mean())
        # #nodice版本
        # l_ncc,l_seg,l_reg  = loss
        # l_tot = (l_ncc.mean() +0.2* l_seg.mean()+ l_reg.mean())
        l_tot.backward()
        self.optG.step()

        self.log_dict['l_ncc'] = l_ncc.mean().item()
        self.log_dict['l_seg'] = 0.2*l_seg.mean().item()
        self.log_dict['l_reg'] = l_reg.mean().item()
        self.log_dict['l_dsc'] = l_dice.mean().item()
        # 记录总的损失到日志
        self.log_dict['l_tot'] = l_tot.item()


    def test_generation(self, continuous=False):
        self.netG.eval()
        input = torch.cat([self.data['M'], self.data['F']], dim=1)
        if isinstance(self.netG, nn.DataParallel):
            self.MF = self.netG.module.generation(input, continuous)
        else:
            self.MF= self.netG.generation(input, continuous)
        self.netG.train()

    def test_registration(self):
        self.netG.eval()
        input = self.data
        if isinstance(self.netG, nn.DataParallel):
            [flowall,warped,warped_mask,seg_out_m,seg_out_f],loss = self.netG.module.registration(input)
        else:
            flowall,warped,warped_mask,seg_out_m,seg_out_f= self.netG.registration(input)
        self.netG.train()
        return flowall,warped,warped_mask,seg_out_m,seg_out_f
    
    def test_registration_origin(self):
        self.netG.eval()
        input = self.data
        if isinstance(self.netG, nn.DataParallel):
            [warped,warped_mask,final_flow],loss = self.netG.module.registration_origin(input)
        else:
            warped,warped_mask,final_flow = self.netG.registration_origin(input)
        self.netG.train()
        return warped,warped_mask,final_flow

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals_train(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['M'] = Metrics.tensor2im(self.data['M'].detach().float().cpu(), min_max=min_max)
        out_dict['F'] = Metrics.tensor2im(self.data['F'].detach().float().cpu(), min_max=min_max)
        out_dict['out_M'] = Metrics.tensor2im(self.out_M.detach().float().cpu(), min_max=(0, 1))
        out_dict['flow'] = Metrics.tensor2im(self.flow.detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['MF'] = Metrics.tensor2im(self.MF.detach().float().cpu(), min_max=min_max)
        out_dict['M'] = Metrics.tensor2im(self.data['M'].detach().float().cpu(), min_max=min_max)
        out_dict['F'] = Metrics.tensor2im(self.data['F'].detach().float().cpu(), min_max=min_max)
        out_dict['out_M'] = Metrics.tensor2im(self.out_M.detach().float().cpu(), min_max=(0, 1))
        out_dict['flow'] = Metrics.tensor2im(self.flow.detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_generation(self):
        out_dict = OrderedDict()

        out_dict['MF'] = self.MF.detach().float().cpu()
        return out_dict

    def get_current_registration(self):
        out_dict = OrderedDict()

        out_dict['out_M'] =self.out_M.detach().float().cpu()
        out_dict['flow'] = self.flow.detach().float().cpu()
        out_dict['contD'] = self.contD.detach().float().cpu()
        out_dict['contF'] = self.contF.detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                             net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)

        logger.info(
            'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(genG_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            genG_path = '{}_gen_G.pth'.format(load_path)

            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                genG_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']