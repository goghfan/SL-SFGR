import torch
import torch.nn as nn
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

if __name__ == '__main__':
    ff = FFParser3D(3,192,160,192)
    x = torch.randn(2, 3, 192, 160, 192)
    y = ff(x)

'''
import torch

def dct3(x):
    """
    三维离散余弦变换
    """
    return torch.fft.fftn(x, dim=(-1, -2, -3)).real

def idct3(y):
    """
    三维逆离散余弦变换
    """
    sizes = y.shape[-3:]
    return torch.fft.ifftn(y, dim=(-1, -2, -3)).real * (2.0 / (sizes[0] * sizes[1] * sizes[2]))

'''