import torch
import torch.nn as nn

class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x


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