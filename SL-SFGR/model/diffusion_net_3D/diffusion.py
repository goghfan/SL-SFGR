import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from model.diffusion_net_3D.loss import crossCorrelation3D
from model.diffusion_net_3D.loss import gradientLoss
from model.diffusion_net_3D.loss import Dice
from model.diffusion_net_3D.loss import DiceLoss
from model.diffusion_net_3D.loss import NCC
import torch.nn.functional as nnf
from model.Models.SwinTranformer3D import TransDecoderBlock,pivit,pivit_origin
# from model.Models.SwinTranformer3D_copy import pivit
from model.diffusion_net_3D.attnunet import AttentionUNet3D as attensionunet
from model.NGF.normalized_gradient_field import NormalizedGradientField3d

import SimpleITK as sitk
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels=6, out_channels=16,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main1 = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.main2 = nn.Conv3d(out_channels, 3, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x,y):
        combine = torch.cat((x,y),dim=1)
        out = self.main1(combine)
        out = self.main2(out)
        out = self.norm(out)
        out = self.activation(out)
        return out

#denoise 是DDPM的U-Net
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        loss_lambda=1,
        trainortest='train'
    ):

        super().__init__()        
        # if trainortest == 'train':
        #     torch.cuda.set_device(4)
        # else:
        #     torch.cuda.set_device(5)
        # if trainortest == 'train':
        #     self.devices = [4,5,6,7]
        # else:
        #     self.devices = [5,6,7]       
        # self.channels = channels
        # self.denoise_fn =  denoise_fn
        # self.seg_unet = attensionunet(1,1) 
        # self.conditional = conditional
        self.loss_type = loss_type
        self.lambda_L = loss_lambda
        # self.encoder =  seg_and_encoder().to(torch.float16) 
        # self.encoder =  seg_and_encoder(num_classes=12)
        # 4 8 16 32 64
        # self.TFB0 =  TransDecoderBlock(4,4,32) 
        # self.TFB1 =  TransDecoderBlock(8,8,32) 
        # self.TFB2 =  TransDecoderBlock(16,16,32) 
        # self.TFB3 =  TransDecoderBlock(32,32,32) 
        # self.TFB4 =  TransDecoderBlock(64,64,32)

        # self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # self.Conv34 =  ConvInsBlock(in_channels=6) 
        # self.Conv12 =  ConvInsBlock(in_channels=6)
        # self.Conv01 =  ConvInsBlock(in_channels=6)  
        # self.Convfinal =  ConvInsBlock(in_channels=6)
        # self.Conv0 =  ConvInsBlock(in_channels=8,out_channels=16)
        # self.softmax = nn.Softmax(dim=1)  
        # self.ffparse = [FFParse(4,192,160,192) , FFParse(8,96,80,96) ]
        # self.pivit = pivit()
        self.pivit = pivit()
        # if schedule_opt is not None:
        #     pass
        # #CT inshape = (96,96,80)

        # #MRI
        # inshape = (96,96,80)
        # self.transformer = nn.ModuleList()
        # for i in range(4):
        #     self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))
        # #MRI
        # self.transformer_final = SpatialTransformer([192,192,160])
        #CT self.transformer_final = SpatialTransformer([192,160,192])

    def p_losses_origin(self, x_in, noise=None):
        #采用pivit作为基础结构的网络 warped_x,flowall,warped_mask,x_seg_m,x_seg_f

        warped,flowall,warped_mask,x_seg_m,x_seg_f= self.pivit(x_in['M'],x_in['F'])
        # l_seg = self.CEL1(seg_out_f,x_in['FL'])+self.CEL2(seg_out_m,x_in['ML'])
        l_ncc = self.loss_ncc.loss(warped,x_in['F'])
        # l_reg = self.loss_reg(flowall)
        # # l_mse = self.loss_mse(warped,x_in['F'])
        
        # # warped_mask = self.softmax(warped_mask)
        
        # l_dice = self.dice(warped_mask,x_in['FL'])
        return l_ncc
        # return l_mse,l_seg,l_dice,l_reg

    def p_losses(self, x_in, noise=None):
        #采用pivit作为基础结构的网络
        warped,flowall,warped_mask,seg_out_m,seg_out_f = self.pivit(x_in)
        l_seg = self.CEL1(seg_out_f,x_in['FL'])+self.CEL2(seg_out_m,x_in['ML'])
        l_ncc = self.loss_ncc.loss(warped,x_in['F'])
        l_reg = self.loss_reg(flowall)
        # l_mse = self.loss_mse(warped,x_in['F'])
        
        # warped_mask = self.softmax(warped_mask)
        
        l_dice = self.dice(warped_mask,x_in['FL'])
        return l_ncc,l_seg,l_dice,l_reg
        # return l_mse,l_seg,l_dice,l_reg        

    def p_losses_nodice(self, x_in, noise=None):
        #采用pivit作为基础结构的网络
        warped,flowall,warped_mask,seg_out_m,seg_out_f = self.pivit(x_in)
        l_seg = self.CEL1(seg_out_f,x_in['FL'])+self.CEL2(seg_out_m,x_in['ML'])
        l_ncc = self.loss_ncc.loss(warped,x_in['F'])
        l_reg = self.loss_reg(flowall)
        # l_mse = self.loss_mse(warped,x_in['F'])
        
        # warped_mask = self.softmax(warped_mask)
        
        # l_dice = self.dice(warped_mask,x_in['FL'])
        return l_ncc,l_seg,l_reg
        # return l_mse,l_seg,l_dice,l_reg       


    def set_loss(self, device):
        # if self.loss_type == 'l1':
        #     self.loss_func = nn.L1Loss(reduction='mean').to(device)
        # elif self.loss_type == 'l2':
        #     self.loss_func = nn.MSELoss(reduction='mean').to(device)
        # else:
        #     raise NotImplementedError()
        #self.loss_ngf = NormalizedGradientField3d(mm_spacing=[1.6,1.6,1.6],reduction='mean').to(device)
        # self.loss_ncc = crossCorrelation3D(1, kernel=(9, 9, 9)).to(device)
        # self.loss_mse = nn.MSELoss(reduction='mean').to(device)
        self.loss_ncc = NCC()
        self.loss_reg = gradientLoss("l2").to(device)
        #二类DICE
        # self.loss_segr = Dice().to(device)
        #多类DICE
        self.dice = DiceLoss(num_classes=12).to(device)
        self.CEL1 = nn.CrossEntropyLoss().to(device)
        self.CEL2 = nn.CrossEntropyLoss().to(device)
        # self.loss_segm = nn.BCELoss().to(device)
        # self.loss_segf = nn.BCELoss().to(device)
    def forward(self, x, *args, **kwargs):
        
        #return self.p_losses_nodice(x, *args, **kwargs)
        return self.p_losses(x, *args, **kwargs)
        # return self.p_losses_origin(x, *args, **kwargs)
    # @torch.no_grad()
    def p_sample_loop(self, x_in):
        '''
        with torch.no_grad():
            M_F,seg_out_m,seg_out_f = self.encoder(x_in)
            # seg_out_m,seg_out_f = torch.split(seg_out,1,dim=0)
            M = []
            F = []
            for temp in M_F:
                M.append(temp[:1, :, :, :, :])
                F.append(temp[1:, :, :, :, :])
            flow4 = self.TFB4(M[4],F[4])
            flow4u = self.upsample_trilin(2*flow4)
            M[3] = self.transformer[3](M[3],flow4u)

            flow3 = self.TFB3(M[3],F[3])
            flow3 = self.Conv34(flow3,flow4u)
            flow3 = self.upsample_trilin(2*flow3)
            M[2] = self.transformer[2](M[2],flow3)
            # 到这里flow2u是1 3 24 20 24


            flow2 = self.TFB2(M[2],F[2])
            flow2 = self.Conv12(flow2,flow3)
            flow2 = self.upsample_trilin(2*flow2)
            M[1] = self.transformer[1](M[1],flow2)
            # 到这里flow2是1 3 48 40 48
            
            flow1 = self.TFB1(M[1],F[1])
            flow1 = self.Conv01(flow1,flow2)
            flow1 = self.upsample_trilin(2*flow1)
            M[0] = self.transformer[0](M[0],flow1)
            # 到这里flow1是1 3 96 80 96

            
            flow0 = self.Conv0(M[0],F[0])
            flow_final = self.Convfinal(flow0,flow1)
            flow_final = self.upsample_trilin(2*flow_final)

            warped = self.transformer_final(x_in['M'],flow_final)

            warped_mask = self.transformer_final(x_in['ML'].unsqueeze(0).float(),flow_final)

            seg_out_m= self.softmax(seg_out_m)
            seg_out_f= self.softmax(seg_out_f)

            return flow_final,warped,warped_mask,seg_out_m,seg_out_f
        '''
        with torch.no_grad():
            #采用pivit作为基础结构的网络
            warped,flowall,warped_mask,seg_out_m,seg_out_f = self.pivit(x_in)
            # l_ncc = self.loss_ncc.loss(warped,x_in['F'])
            # l_mse = self.loss_mse(warped,x_in['F'])
            # l_seg = self.CEL1(seg_out_f,x_in['FL'])+self.CEL2(seg_out_m,x_in['ML'])
            # warped_mask = self.softmax(warped_mask)
            # l_dice = self.dice(warped_mask,x_in['FL'])
            # l_reg = self.loss_reg(flowall)
            return flowall,warped,warped_mask,seg_out_m,seg_out_f

    def p_sample_loop_pivit(self, x_in):
        '''
        with torch.no_grad():
            M_F,seg_out_m,seg_out_f = self.encoder(x_in)
            # seg_out_m,seg_out_f = torch.split(seg_out,1,dim=0)
            M = []
            F = []
            for temp in M_F:
                M.append(temp[:1, :, :, :, :])
                F.append(temp[1:, :, :, :, :])
            flow4 = self.TFB4(M[4],F[4])
            flow4u = self.upsample_trilin(2*flow4)
            M[3] = self.transformer[3](M[3],flow4u)

            flow3 = self.TFB3(M[3],F[3])
            flow3 = self.Conv34(flow3,flow4u)
            flow3 = self.upsample_trilin(2*flow3)
            M[2] = self.transformer[2](M[2],flow3)
            # 到这里flow2u是1 3 24 20 24


            flow2 = self.TFB2(M[2],F[2])
            flow2 = self.Conv12(flow2,flow3)
            flow2 = self.upsample_trilin(2*flow2)
            M[1] = self.transformer[1](M[1],flow2)
            # 到这里flow2是1 3 48 40 48
            
            flow1 = self.TFB1(M[1],F[1])
            flow1 = self.Conv01(flow1,flow2)
            flow1 = self.upsample_trilin(2*flow1)
            M[0] = self.transformer[0](M[0],flow1)
            # 到这里flow1是1 3 96 80 96

            
            flow0 = self.Conv0(M[0],F[0])
            flow_final = self.Convfinal(flow0,flow1)
            flow_final = self.upsample_trilin(2*flow_final)

            warped = self.transformer_final(x_in['M'],flow_final)

            warped_mask = self.transformer_final(x_in['ML'].unsqueeze(0).float(),flow_final)

            seg_out_m= self.softmax(seg_out_m)
            seg_out_f= self.softmax(seg_out_f)

            return flow_final,warped,warped_mask,seg_out_m,seg_out_f
        '''
        with torch.no_grad():
            #采用pivit作为基础结构的网络
            warped,flowall = self.pivit(x_in['M'],x_in['F'])
            transformer_final = SpatialTransformer((192,192,160)).to(flowall.device)
            warped_mask = transformer_final(x_in['ML'].unsqueeze(0).float(),flowall)

            return warped,warped_mask,flowall

    def decode_one_hot(self,one_hot_map):
        """
        Decode one-hot representation into label map.

        Args:
        - one_hot_map: The one-hot representation of the label map with shape [depth, height, width, num_classes].

        Returns:
        - label_map: The decoded label map with shape [depth, height, width].
        """
        one_hot_map.to('cpu')
        label_map = np.argmax(one_hot_map, axis=-1)
        return label_map

    @torch.no_grad()
    def registration(self, x_in):
        # return self.p_sample_loop_pivit(x_in)
        return self.p_sample_loop(x_in)

    @torch.no_grad()
    def registration_origin(self, x_in):
        # return self.p_sample_loop_pivit(x_in)
        return self.p_sample_loop_pivit(x_in)

    # def q_sample(self, x_start, t, noise=None):
    #     noise = default(noise, lambda: torch.randn_like(x_start))
    #     # fix gama
    #     return (
    #         extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
    #         extract(self.sqrt_one_minus_alphas_cumprod,
    #                 t, x_start.shape) * noise
    #     )

