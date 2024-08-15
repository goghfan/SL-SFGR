'''
This is the first encoder which can extract the features of Moving Images and Fixed Images.
CWM is the adaptive weight module. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.normal import Normal
from model.diffusion_net_3D.attnunet import GridAttentionBlock3D
from model.bidirectional_attention import CrossAttention3DModule
from model.Models.FFparse import FFParser3D
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

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        x = x[:,:,1:-1,1:-1,1:-1]
        return self.actout(x)

class DeconvBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels):
        super(DeconvBlock, self).__init__()
        self.upconv = UpConvBlock(dec_channels, skip_channels)
        self.conv = nn.Sequential(
            ConvInsBlock(2*skip_channels, skip_channels),
            ConvInsBlock(skip_channels, skip_channels)
        )
    def forward(self, dec, skip):
        dec = self.upconv(dec)
        out = self.conv(torch.cat([dec, skip], dim=1))
        return out

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, c),
            ConvInsBlock(c, c)
        )
        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c)
        )
        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2*c, 4*c),
            ConvInsBlock(4*c, 4*c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4*c, 8*c),
            ConvInsBlock(8*c, 8*c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8*c, 16*c),
            ConvInsBlock(16*c, 16*c)
        )

        # self.conv4 = nn.Sequential(
        #     nn.AvgPool3d(2),
        #     ConvInsBlock(16 * c, 32 * c),
        #     ConvInsBlock(32 * c, 32 * c)
        # )

    def forward(self, x):
        x=x.to(torch.float16)
        out0 = self.conv0(x)  # 1           4     192 160 192
        out1 = self.conv1(out0)  # 1/2      8     96   80  96
        out2 = self.conv2(out1)  # 1/4      16    48   40  48
        out3 = self.conv3(out2)  # 1/8      32    24   20  24
        out4 = self.conv4(out3)  # 1/8      64    12   10   12

        return [out0.to(torch.float32), out1.to(torch.float32), out2.to(torch.float32), out3.to(torch.float32), out4.to(torch.float32)]
    

# class CWM(nn.Module):
#     def __init__(self, in_channels, channels):
#         super(CWM, self).__init__()

#         c = channels
#         self.num_fields = in_channels // 3

#         self.conv = nn.Sequential(
#             ConvInsBlock(in_channels, channels, 3, 1),
#             ConvInsBlock(channels, channels, 3, 1),
#             nn.Conv3d(channels, self.num_fields, 3, 1, 1),
#             nn.Softmax(dim=1)
#         )

#         self.upsample = nn.Upsample(
#                 scale_factor=2,
#                 mode='trilinear',
#                 align_corners=True
#             )

#     def forward(self, x):

#         x = self.upsample(x)
#         weight = self.conv(x)

#         weighted_field = 0

#         for i in range(self.num_fields):
#             w = x[:, 3*i: 3*(i+1)]
#             weight_map = weight[:, i:(i+1)]
#             weighted_field = weighted_field + w*weight_map

#         return 2*weighted_field

# class seg_and_encoder(nn.Module):
#     def __init__(self,num_classes,in_channel=1, first_out_channel=4):
#         super(seg_and_encoder,self).__init__()
#         c = first_out_channel

#         self.conv0 = nn.Sequential(
#             ConvBlock(in_channel, c),
#             ConvInsBlock(c, c),
#             ConvInsBlock(c, c),
#             nn.AvgPool3d(2)
#         )
#         self.conv1 = nn.Sequential(
#             ConvInsBlock(c, 2*c),
#             ConvInsBlock(2*c, 2*c),
#             nn.AvgPool3d(2)
#         )
#         self.conv2 = nn.Sequential(
#             ConvInsBlock(2*c, 4*c),
#             ConvInsBlock(4*c, 4*c),
#             nn.AvgPool3d(2)
#         )

#         self.conv3 = nn.Sequential(
#             ConvInsBlock(4*c, 8*c),
#             ConvInsBlock(8*c, 8*c),
#             nn.AvgPool3d(2)
#         )

#         self.conv4 = nn.Sequential(
#             ConvInsBlock(8*c, 16*c),
#             ConvInsBlock(16*c, 16*c),
#             nn.AvgPool3d(2)
#         )

#         self.encoderconv1 = UnetConv3(in_channel, 4)
#         self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         #---------------------------------------------------
#         self.encoderconv2 = UnetConv3(4, 8)
#         self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         #---------------------------------------------------
#         self.encoderconv3 = UnetConv3(8, 16)
#         self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         #---------------------------------------------------
#         self.encoderconv4 = UnetConv3(16, 32)
#         self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         #---------------------------------------------------
#         self.encoderconv5 = UnetConv3(32, 64)
#         self.maxpool5 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         #---------------------------------------------------        
#         self.center = UnetConv3(64, 128)
#         self.gating = UnetGridGatingSignal(128, 64, kernel_size=(1, 1, 1))

#         self.attentionblock5 = GridAttentionBlock3D(64, 64)
#         self.attentionblock4 = GridAttentionBlock3D(32, 64)
#         self.attentionblock3 = GridAttentionBlock3D(16, 64)
#         self.attentionblock2 = GridAttentionBlock3D(8, 64)

#         self.up_concat5 = UnetUp3(128, 64)
#         self.up_concat4 = UnetUp3(64, 32)
#         self.up_concat3 = UnetUp3(32, 16)
#         self.up_concat2 = UnetUp3(16, 8)
#         self.up_concat1 = UnetUp3(8, 4)

#         self.out_conv = nn.Conv3d(4, num_classes, 1)
#         self.sigmod= nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)

#         self.cross_att1 = CrossAttention3DModule(4).to(torch.float32) 
#         self.cross_att2 = CrossAttention3DModule(8).to(torch.float32)  
#         self.ffparse1 = FFParser3D(4,96,80,96).to(torch.float32) 
#         self.ffparse2 = FFParser3D(8,48,40,48).to(torch.float32) 
 
#     def forward(self, x_in):
#         #input = torch.cat([x_in['M'],x_in['F']],dim=0).to(torch.float16)
#         input = torch.cat([x_in['M'],x_in['F']],dim=0)
#         #input_seg = torch.cat([x_in['M']*x_in['ML'],x_in['F']*x_in['FL']],dim=0)

#         x_en1 = self.encoderconv1(input)
#         pool1 = self.maxpool1(x_en1)
#         out0 = self.conv0(input)
#         # pool1,out0 = self.ffparse1(torch.cat([pool1,out0],dim=0))
#         # pool1,out0 = self.cross_att1(pool1,out0)

       
#         x_en2 = self.encoderconv2(pool1)
#         pool2 = self.maxpool2(x_en2)
#         out1 = self.conv1(out0)
#         # pool2,out1 = self.ffparse2(torch.cat([pool2,out1],dim=0))  
#         # pool2,out1 = self.cross_att2(pool2 ,out1)
        

#         out2 = self.conv2(out1)  # 1/4      16    48   40  48
#         out3 = self.conv3(out2)  # 1/8      32    24   20  24
#         out4 = self.conv4(out3)  # 1/8      64    12   10   12
        
#         x_en3 = self.encoderconv3(pool2)
#         pool3 = self.maxpool3(x_en3)

#         x_en4 = self.encoderconv4(pool3)
#         pool4 = self.maxpool4(x_en4)

#         x_en5 = self.encoderconv5(pool4)
#         pool5 = self.maxpool5(x_en5)

#         center = self.center(pool5)
#         gating = self.gating(center)

#         att5 = self.attentionblock5(x_en5, gating)
#         att4 = self.attentionblock4(x_en4, gating)
#         att3 = self.attentionblock3(x_en3, gating)
#         att2 = self.attentionblock2(x_en2, gating)

#         up5 = self.up_concat5(att5, center)
#         up4 = self.up_concat4(att4, up5)
#         up3 = self.up_concat3(att3, up4)
#         up2 = self.up_concat2(att2, up3)
#         up1 = self.up_concat1(x_en1, up2)

#         x = self.out_conv(up1)
        
#         # x = self.sigmod(x)
#         # x = (self.sigmod(x)>0.5).float()
#         x_seg_m,x_seg_f = torch.split(x,1,dim=0)
#         # x_seg_m= self.softmax(x_seg_m)
#         # x_seg_f= self.softmax(x_seg_f)
        
#         return [out0.to(torch.float32),out1.to(torch.float32), out2.to(torch.float32), out3.to(torch.float32), out4.to(torch.float32)],x_seg_m.to(torch.float32),x_seg_f.to(torch.float32)

if __name__ == '__main__':
    x = {'M':torch.randn(1,1,160,192,192).to('cuda:4'),'F':torch.randn(1,1,160,192,192).to('cuda:4')}
    unet = seg_and_encoder(num_classes=12).to('cuda:4')
    y = unet(x)