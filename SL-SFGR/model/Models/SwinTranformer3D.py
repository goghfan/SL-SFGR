import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, trunc_normal_

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from model.diffusion_net_3D.attnunet import GridAttentionBlock3D, DecoderConv, UpsampleAndConcat



class CrossAttention3DModule(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention3DModule, self).__init__()

        # Assuming in_channels is the number of input channels for A and B
        self.conv_query_a = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_key_b = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_value_b = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.conv_query_b = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_key_a = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv_value_a = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, input_a, input_b):
        # Calculate attention weights when A is query and B is key-value
        query_a = self.conv_query_a(input_a)
        key_b = self.conv_key_b(input_b)
        value_b = self.conv_value_b(input_b)

        attention_weights_a_to_b = F.softmax(torch.matmul(query_a.view(query_a.size(0), -1), key_b.view(key_b.size(0), -1).T), dim=-1)
        output_a_to_b = torch.matmul(attention_weights_a_to_b, value_b.view(value_b.size(0), -1)).view(value_b.size())

        # Calculate attention weights when B is query and A is key-value
        query_b = self.conv_query_b(input_b)
        key_a = self.conv_key_a(input_a)
        value_a = self.conv_value_a(input_a)

        attention_weights_b_to_a = F.softmax(torch.matmul(query_b.view(query_b.size(0), -1), key_a.view(key_a.size(0), -1).T), dim=-1)
        output_b_to_a = torch.matmul(attention_weights_b_to_a, value_a.view(value_a.size(0), -1)).view(value_a.size())

        return output_b_to_a.to(torch.float32),output_a_to_b.to(torch.float32) 

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class FFParser3D(nn.Module):
    def __init__(self, dim, d, h, w):
        super().__init__()
        # 定义一个复杂权重参数，形状为 (dim, d, h, w, 2)
        self.complex_weight = nn.Parameter(torch.randn(dim, d, h, w, 2, dtype=torch.float32) * 0.02).to('cuda')

    def forward(self, x):
        B, C, D, H, W = x.shape

        # 将输入张量 x 转换为 float32 类型
        x = x.to(torch.float32)
        
        # 对输入张量进行三维快速傅立叶变换
        x = torch.fft.fftn(x, dim=(2, 3, 4), norm='ortho')
        
        # 将复杂权重参数视为复数形式
        weight = torch.view_as_complex(self.complex_weight.to(x.device))
        weight = torch.unsqueeze(weight, dim=0)
        
        # 在频域上应用复杂权重
        x = x * weight
        
        # 对输入张量进行三维逆傅立叶变换
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(2, 3, 4), norm='ortho')

        # 重新调整张量的形状
        # 返回前两个维度和后两个维度的张量
        x_first_two = x[:B//2, :, :, :, :]
        x_last_two = x[B//2:, :, :, :, :]

        return x_first_two , x_last_two

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):

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

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x





class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=32):
        super(Encoder, self).__init__()
        c = first_channel
        self.block1 = ConvBlock(in_channel, c, stride=2)
        self.block2 = ConvBlock(c, c , stride=2)
        self.block3 = ConvBlock(c, c, stride=2)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        return out1, out2, out3

class Seg_Reg_Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=32):
        super(Seg_Reg_Encoder, self).__init__()
        c = first_channel
        # self.seg_block1 = ConvBlock(in_channel, c, stride=2)
        # self.seg_block2 = ConvBlock(c, c , stride=2)
        # self.seg_block3 = ConvBlock(c, c, stride=2)
        # Encoder layers
        self.encoder1 = EncoderConv(in_channel, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = EncoderConv(32, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = EncoderConv(32, 32)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.reg_block1 = ConvBlock(in_channel, c, stride=2)
        self.reg_block2 = ConvBlock(c, c , stride=2)
        self.reg_block3 = ConvBlock(c, c, stride=2)

        self.cross1 = CrossAttention3DModule(32)
        self.cross2 = CrossAttention3DModule(32)
        self.cross3 = CrossAttention3DModule(32)
    def forward(self,x_in):
        input = torch.cat([x_in['M'],x_in['F']],dim=0)
        out_seg1 = self.encoder1(input)
        out_seg1_pool1 = self.pool1(out_seg1)
        out_reg1 = self.reg_block1(input)
        out_seg1_pool1,out_reg1 = self.cross1(out_seg1_pool1,out_reg1)

        out_seg2 = self.encoder2(out_seg1_pool1)
        out_seg2_pool2 = self.pool2(out_seg2)
        out_reg2 = self.reg_block2(out_reg1)
        out_seg2_pool2,out_reg2 = self.cross2(out_seg2_pool2,out_reg2)

        out_seg3 = self.encoder3(out_seg2_pool2)
        out_seg3_pool3 = self.pool3(out_seg3)
        out_reg3 = self.reg_block3(out_reg2)
        out_seg3_pool3,out_reg3 = self.cross3(out_seg3_pool3,out_reg3)

        return out_seg2,out_seg3,out_seg3_pool3,out_reg1,out_reg2,out_reg3

class Seg_Reg_Encoder_noatt(nn.Module):
    def __init__(self, in_channel=1, first_channel=32):
        super(Seg_Reg_Encoder_noatt, self).__init__()
        c = first_channel
        # self.seg_block1 = ConvBlock(in_channel, c, stride=2)
        # self.seg_block2 = ConvBlock(c, c , stride=2)
        # self.seg_block3 = ConvBlock(c, c, stride=2)
        # Encoder layers
        self.encoder1 = EncoderConv(in_channel, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = EncoderConv(32, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = EncoderConv(32, 32)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.reg_block1 = ConvBlock(in_channel, c, stride=2)
        self.reg_block2 = ConvBlock(c, c , stride=2)
        self.reg_block3 = ConvBlock(c, c, stride=2)

        # self.cross1 = CrossAttention3DModule(32)
        # self.cross2 = CrossAttention3DModule(32)
        # self.cross3 = CrossAttention3DModule(32)
    def forward(self,x_in):
        input = torch.cat([x_in['M'],x_in['F']],dim=0)
        out_seg1 = self.encoder1(input)
        out_seg1_pool1 = self.pool1(out_seg1)
        out_reg1 = self.reg_block1(input)
        # out_seg1_pool1,out_reg1 = self.cross1(out_seg1_pool1,out_reg1)

        out_seg2 = self.encoder2(out_seg1_pool1)
        out_seg2_pool2 = self.pool2(out_seg2)
        out_reg2 = self.reg_block2(out_reg1)
        # out_seg2_pool2,out_reg2 = self.cross2(out_seg2_pool2,out_reg2)

        out_seg3 = self.encoder3(out_seg2_pool2)
        out_seg3_pool3 = self.pool3(out_seg3)
        out_reg3 = self.reg_block3(out_reg2)
        # out_seg3_pool3,out_reg3 = self.cross3(out_seg3_pool3,out_reg3)

        return out_seg2,out_seg3,out_seg3_pool3,out_reg1,out_reg2,out_reg3

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads=6,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])  # 4

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class TransDecoderBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel):
        super(TransDecoderBlock, self).__init__()
        self.Swin1 = BasicLayer(dim = x_channel + y_channel, depth =2, num_heads = 1, window_size=(8, 8, 8))
        self.Conv2 = ConvBlock(x_channel + y_channel, out_channel)
        self.Conv3 = nn.Conv3d(out_channel, 3, 3, padding=1)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        x = self.Swin1(concat)
        x = self.Conv2(x)
        flow = self.Conv3(x)
        return flow

class DecoderBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.Conv1 = ConvBlock(x_channel + y_channel, out_channel)
        self.Conv2 = nn.Conv3d(out_channel, 3, 3, padding=1)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        x = self.Conv1(concat)
        flow = self.Conv2(x)
        return flow


class pivit(nn.Module):
    # @store_config_args
    def __init__(self, size=(192,192,160), in_channel=1, first_channel=32):
        super(pivit, self).__init__()
        c = first_channel
        self.seg_reg_encoder = Seg_Reg_Encoder(in_channel, c)
        self.decoder31 = TransDecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder32 = TransDecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder33 = TransDecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder2 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder1 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.size = size
        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        #seg_network
        #---------------------------------------------------
        self.encoder4 = EncoderConv(32, 64)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder5 = EncoderConv(64, 128)
        
        # Attention layers
        # self.attention2 = GridAttentionBlock3D(32, 32)
        # self.attention3 = GridAttentionBlock3D(32, 32)
        # self.attention4 = GridAttentionBlock3D(64, 64)
        
        # Decoder layers
        self.decoder11 = DecoderConv(128 + 64, 64)
        self.up_and_concat1 = UpsampleAndConcat(128, 64)
        
        self.decoder22 = DecoderConv(64 + 32, 32)
        self.up_and_concat2 = UpsampleAndConcat(64, 32)
        
        self.decoder333 = DecoderConv(32 + 32, 12)
        self.up_and_concat3 = UpsampleAndConcat(32, 32)
                
        # Output layer
        self.out_conv = nn.Conv3d(12, 12, kernel_size=1)   



    def forward(self, x_in):               
        # fx1, fx2, fx3 = self.encoder(x)
        # fy1, fy2, fy3 = self.encoder(y)
        # x_en1 = self.encoderconv1(input)
        # x_en2 = self.encoderconv2(x_en1)
        # x_en3 = self.encoderconv3(x_en2)
        e2,e3,e3_pool,fxy1,fxy2,fxy3=self.seg_reg_encoder(x_in)

        fx1,fy1 = torch.split(fxy1,1,dim=0)
        fx2,fy2 = torch.split(fxy2,1,dim=0)
        fx3,fy3 = torch.split(fxy3,1,dim=0)
        wx3 = fx3

        flow = self.decoder31(wx3, fy3)
        flowall = flow
        wx3 = self.transformer[3](fx3, flowall)

        flow = self.decoder32(wx3, fy3)
        flowall = self.transformer[3](flowall, flow) + flow
        wx3 = self.transformer[3](fx3, flowall)

        flow = self.decoder33(wx3, fy3)
        flowall = self.transformer[3](flowall, flow) + flow

        flowall = self.up(2 * flowall)

        wx2 = self.transformer[2](fx2, flowall)
        flow = self.decoder2(wx2, fy2)
        flowall = self.transformer[2](flowall, flow) + flow
        flowall = self.up(2 * flowall)

        wx1 = self.transformer[1](fx1, flowall)
        flow = self.decoder1(wx1, fy1)
        flowall = self.transformer[1](flowall, flow) + flow
        flowall = self.up(2 * flowall)
        warped_x = self.transformer[0](x_in['M'], flowall)
        warped_mask = self.transformer[0](x_in['ML'].unsqueeze(0).float(), flowall)

        e4 = self.encoder4(e3_pool)
        e4_pool = self.pool4(e4)
        e5 = self.encoder5(e4_pool)
        
        # Attention
        # e2_attention = self.attention2(e2, e2)
        # e3_attention = self.attention3(e3, e3)
        # e4_attention = self.attention4(e4, e4)
        
        # Decoder
        d5 = self.up_and_concat1(e5, e4)
        d1 = self.decoder11(d5)
        
        d2 = self.up_and_concat2(d1, e3)
        d2 = self.decoder22(d2)
        
        d3 = self.up_and_concat3(d2, e2)
        d3 = self.decoder333(d3)
        
        # d4 = self.decoder44(d3)
        
        # Upsample d4
        d3 = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=True)

        # Output
        out = self.out_conv(d3)
        
        x_seg_m,x_seg_f = torch.split(out,1,dim=0)

        return warped_x,flowall,warped_mask,x_seg_m,x_seg_f

class pivit_origin(nn.Module):
    # @store_config_args
    def __init__(self, size=(192,192,160), in_channel=1, first_channel=32):
        super(pivit_origin, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder31 = TransDecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder32 = TransDecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder33 = TransDecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder2 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder1 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.size = size

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, y):
        fx1, fx2, fx3 = self.encoder(x)
        fy1, fy2, fy3 = self.encoder(y)

        wx3 = fx3

        flow = self.decoder31(wx3, fy3)
        flowall = flow
        wx3 = self.transformer[3](fx3, flowall)

        flow = self.decoder32(wx3, fy3)
        flowall = self.transformer[3](flowall, flow) + flow
        wx3 = self.transformer[3](fx3, flowall)

        flow = self.decoder33(wx3, fy3)
        flowall = self.transformer[3](flowall, flow) + flow

        flowall = self.up(2 * flowall)

        wx2 = self.transformer[2](fx2, flowall)
        flow = self.decoder2(wx2, fy2)
        flowall = self.transformer[2](flowall, flow) + flow
        flowall = self.up(2 * flowall)

        wx1 = self.transformer[1](fx1, flowall)
        flow = self.decoder1(wx1, fy1)
        flowall = self.transformer[1](flowall, flow) + flow
        flowall = self.up(2 * flowall)

        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall

if __name__ == '__main__':
    initial_memory = torch.cuda.memory_allocated()
    print(f"初始显存使用: {initial_memory / (1024 ** 3):.2f} GB")
    x = torch.randn(1,1,192,192,160).to('cuda')
    y = torch.randn(1,1,192,192,160).to('cuda')
    tra = pivit().to('cuda')
    flows = tra(x, y)
    memory_used = torch.cuda.memory_allocated() - initial_memory
    print(f"运行后显存使用: {memory_used / (1024 ** 3):.2f} GB")