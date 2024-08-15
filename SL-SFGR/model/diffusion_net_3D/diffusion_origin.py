import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from model.diffusion_net_3D.loss import crossCorrelation3D
from model.diffusion_net_3D.loss import gradientLoss
import torch.nn.functional as nnf
from model.Models.Encoder import Encoder
from model.Models.SwinTranformer3D import TransDecoderBlock
from model.Models.FFparse import FFParser3D as FFParse
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

    def __init__(self, in_channels=6, out_channels=32,kernal_size=3, stride=1, padding=1, alpha=0.1):
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


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()



#denoise 是DDPM的U-Net
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        loss_lambda=1
    ):
        torch.cuda.set_device(4)
        super().__init__()
        self.devices = [4,5,6,7]
        self.channels = channels
        self.denoise_fn =  denoise_fn 
        self.conditional = conditional
        self.loss_type = loss_type
        self.lambda_L = loss_lambda
        self.encoder =  Encoder() 
        
        # 4 8 16 32 64
        self.TFB0 =  TransDecoderBlock(8,4,32) 
        self.TFB1 =  TransDecoderBlock(16,8,32) 
        self.TFB2 =  TransDecoderBlock(32,16,32) 
        self.TFB3 =  TransDecoderBlock(64,32,32) 
        self.TFB4 =  TransDecoderBlock(128,64,32) 
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.cwm =  ConvInsBlock() 

        self.ffparse = [ FFParse(4,192,160,192) , FFParse(8,96,80,96) ]
        if schedule_opt is not None:
            pass
        inshape = (192,160,192)
        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append( SpatialTransformer([s // 2**i for s in inshape]) )
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError()
        self.loss_ncc = crossCorrelation3D(1, kernel=(9, 9, 9)).to(device)
        self.loss_reg = gradientLoss("l2").to(device)


    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        if condition_x is not None:
            with torch.no_grad():
                score = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)

            x_recon = self.predict_start_from_noise(
                x, t=t, noise=score)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample_loop(self, x_in, nsample, continous=False):
        device = self.betas.device
        x = x_in
        x_m = x_in[:, :1]
        x_f = x_in[:, 1:]
        b, c, d, h, w = x_m.shape

        with torch.no_grad():
            t = torch.full((b,), 0, device=device, dtype=torch.long)
            score = self.denoise_fn(torch.cat([x, x_f], dim=1), t)
            #denoise 是DDPM的U-Net
            #fleid   是VM的U-Net
            gamma = np.linspace(0, 1, nsample)
            b, c, d, h, w = x_f.shape
            flow_stack = torch.zeros([1, 3, d, h, w], device=device)
            code_stack = score
            defm_stack = x_m

            for i in (gamma):
                print('-------- Deform with eta=%.3f' % i)
                code = score * (i)
                deform, flow = self.field_fn(torch.cat([x_m, code], dim=1))
                code_stack = torch.cat([code_stack, code], dim=0)
                defm_stack = torch.cat([defm_stack, deform], dim=0)
                flow_stack = torch.cat([flow_stack, flow], dim=0)

        if continous:
            return deform, flow, defm_stack[1:], flow_stack[1:]
        else:
            return deform, flow, defm_stack[-1], flow_stack[-1]

    @torch.no_grad()
    def registration(self, x_in, nsample=7, continuous=False):
        return self.p_sample_loop(x_in, nsample, continuous)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['M']
        [b, c, d, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        M = self.encoder(x_in['M'])
        F = self.encoder(x_in['F'])
        shape_list = [s.shape for s in F]    
        #获取DDPM获得的所有特征
        S,x_recon = self.denoise_fn(torch.cat([x_in['M'], x_in['F'], x_noisy], dim=1), t,shape_list)
        #在FFParse里面写一个循环，循环处理五个特征，以列表返回五个特征
        for index in range(2):
            M[index] = self.ffparse[index](M[index])
            F[index] = self.ffparse[index](F[index])
            S[index] = self.ffparse[index](S[index])
        
        flow4 = self.TFB4(torch.cat([M[4],S[4]],dim=1),F[4])
        M[3] = self.transformer[3](M[3],self.upsample_trilin(2*flow4))

        flow3 = self.TFB3(torch.cat([M[3],S[3]],dim=1),F[3])
        M[2] = self.transformer[2](M[2],self.upsample_trilin(2*flow3))

        flow2 = self.TFB2(torch.cat([M[2],S[2]],dim=1),F[2])
        M[1] = self.transformer[1](M[1],self.upsample_trilin(2*flow2))

        flow1 = self.TFB1(torch.cat([M[1],S[1]],dim=1),F[1])
        flow01 =  self.upsample_trilin(2*flow1)
        M[0] = self.transformer[0](M[0],flow01)
        
        flow02 = self.TFB0(torch.cat([M[0],S[0]],dim=1),F[0])

        flow_final = self.cwm(flow02,flow01)
        warped = self.transformer[0](M[0],flow_final)
        
        #这里再改一下，不要for，单独写
        #损失函数尚未定义，TFB1的顺序是错的，flow的顺序要再改
        #STN需要添加 
        l_diff = self.loss_func(noise, x_recon)
        l_ncc = self.loss_ncc(warped,x_in['F'])
        l_reg = self.loss_reg(flow_final)
        return [x_recon,l_diff,l_ncc,l_reg],warped

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
