# import math
# import torch
# from torch import nn, cat, add
# import numpy as np
# import torch.nn.functional as F

# # UNet3D

# class EncoderConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(EncoderConv, self).__init__()

#         self.conv1 = nn.Conv3d(in_ch, out_ch // 2, 3, padding=1)
#         self.conv2 = nn.Conv3d(out_ch // 2, out_ch, 3, padding=1)

#         self.bn1 = nn.GroupNorm(4, out_ch // 2)
#         self.bn2 = nn.GroupNorm(4, out_ch)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         return x


# class DecoderConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DecoderConv, self).__init__()

#         self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
#         self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

#         self.bn = nn.GroupNorm(4, out_ch)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.relu(x)

#         return x


# class UNet3D(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet3D, self).__init__()
#         self.encoderconv1 = EncoderConv(n_channels, 64)

#         self.encoderconv2 = EncoderConv(64, 128)

#         self.encoderconv3 = EncoderConv(128, 256)

#         self.encoderconv4 = EncoderConv(256, 512)
#         self.up1 = nn.ConvTranspose3d(512, 512, 2, stride=2, padding=0)
#         self.decoderconv1 = DecoderConv(768, 256)
#         self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2, padding=0)
#         self.decoderconv2 = DecoderConv(384, 128)
#         self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2, padding=0)
#         self.decoderconv3 = DecoderConv(192, 64)

#         self.out_conv = nn.Conv3d(64, n_classes, 1)
#         self.maxpooling = nn.MaxPool3d(2)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x_input):
#         x_en1 = self.encoderconv1(x_input)
#         x = self.maxpooling(x_en1)
#         x_en2 = self.encoderconv2(x)
#         x = self.maxpooling(x_en2)
#         x_en3 = self.encoderconv3(x)
#         x = self.maxpooling(x_en3)
#         x_en4 = self.encoderconv4(x)
#         x = self.up1(x_en4)
#         x_de1 = self.decoderconv1(cat([x_en3, x], dim=1))
#         x = self.up2(x_de1)
#         x_de2 = self.decoderconv2(cat([x_en2, x], dim=1))
#         x = self.up3(x_de2)
#         x_de3 = self.decoderconv3(cat([x_en1, x], dim=1))

#         x = self.out_conv(x_de3)

#         x = self.softmax(x)
#         return x


# # Attention UNet3D

# class UnetConv3(nn.Module):
#     def __init__(self, in_size, out_size, kernel_size=(3, 3, 3), padding_size=(1, 1, 1), init_stride=(1, 1, 1)):
#         super(UnetConv3, self).__init__()

#         self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
#                                    nn.GroupNorm(4, out_size),
#                                    nn.ReLU(inplace=True), )
#         self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
#                                    nn.GroupNorm(4, out_size),
#                                    nn.ReLU(inplace=True), )

#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         outputs = self.conv2(outputs)
#         return outputs


# class UnetGridGatingSignal(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=(1, 1, 1)):
#         super(UnetGridGatingSignal, self).__init__()

#         self.conv1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, (1, 1, 1), (0, 0, 0)),
#                                    nn.GroupNorm(4, out_ch),
#                                    nn.ReLU(inplace=True))

#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs


# class GridAttentionBlock3D(nn.Module):
#     def __init__(self, x_ch, g_ch, sub_sample_factor=(2, 2, 2)):
#         super(GridAttentionBlock3D, self).__init__()

#         self.W = nn.Sequential(
#             nn.Conv3d(x_ch,
#                       x_ch,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0),
#             nn.GroupNorm(4, x_ch))
#         self.theta = nn.Conv3d(x_ch,
#                                x_ch,
#                                kernel_size=sub_sample_factor,
#                                stride=sub_sample_factor,
#                                padding=0,
#                                bias=False)
#         self.phi = nn.Conv3d(g_ch,
#                              x_ch,
#                              kernel_size=1,
#                              stride=1,
#                              padding=0,
#                              bias=True)
#         self.psi = nn.Conv3d(x_ch,
#                              out_channels=1,
#                              kernel_size=1,
#                              stride=1,
#                              padding=0,
#                              bias=True)

#     def forward(self, x, g):
#         input_size = x.size()
#         batch_size = input_size[0]
#         assert batch_size == g.size(0)

#         theta_x = self.theta(x)
#         theta_x_size = theta_x.size()

#         phi_g = F.upsample(self.phi(g),
#                            size=theta_x_size[2:],
#                            mode='trilinear')

#         f = F.relu(theta_x + phi_g, inplace=True)

#         sigm_psi_f = F.sigmoid(self.psi(f))
#         sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='trilinear')
#         y = sigm_psi_f.expand_as(x) * x
#         W_y = self.W(y)
#         return W_y


# class UnetUp3(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(UnetUp3, self).__init__()

#         self.conv = UnetConv3(in_size, out_size)
#         self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

#     def forward(self, inputs1, inputs2):
#         outputs2 = self.up(inputs2)
#         offset = outputs2.size()[2] - inputs1.size()[2]
#         padding = 2 * [offset // 2, offset // 2, 0]
#         outputs1 = F.pad(inputs1, padding)
#         return self.conv(torch.cat([outputs1, outputs2], 1))


# class AttentionUNet3D(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(AttentionUNet3D, self).__init__()

#         self.encoderconv1 = UnetConv3(n_channels, 4)
#         self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.encoderconv2 = UnetConv3(4, 8)
#         self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.encoderconv3 = UnetConv3(8, 16)
#         self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.encoderconv4 = UnetConv3(16, 32)
#         self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.center = UnetConv3(32, 64)
#         self.gating = UnetGridGatingSignal(64, 32, kernel_size=(1, 1, 1))

#         self.attentionblock4 = GridAttentionBlock3D(32, 32)
#         self.attentionblock3 = GridAttentionBlock3D(16, 32)
#         self.attentionblock2 = GridAttentionBlock3D(8, 32)

#         self.up_concat4 = UnetUp3(64, 32)
#         self.up_concat3 = UnetUp3(32, 16)
#         self.up_concat2 = UnetUp3(16, 8)
#         self.up_concat1 = UnetUp3(8, 4)

#         self.out_conv = nn.Conv3d(4, n_classes, 1)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x_input):
#         x_en1 = self.encoderconv1(x_input)
#         pool1 = self.maxpool1(x_en1)

#         x_en2 = self.encoderconv2(pool1)
#         pool2 = self.maxpool2(x_en2)

#         x_en3 = self.encoderconv3(pool2)
#         pool3 = self.maxpool3(x_en3)

#         x_en4 = self.encoderconv4(pool3)
#         pool4 = self.maxpool4(x_en4)

#         center = self.center(pool4)
#         gating = self.gating(center)

#         att4 = self.attentionblock4(x_en4, gating)
#         att3 = self.attentionblock3(x_en3, gating)
#         att2 = self.attentionblock2(x_en2, gating)

#         up4 = self.up_concat4(att4, center)
#         up3 = self.up_concat3(att3, up4)
#         up2 = self.up_concat2(att2, up3)
#         up1 = self.up_concat1(x_en1, up2)

#         x = self.out_conv(up1)

#         x = self.softmax(x)
#         return x

# # test
# #
# # from torchsummary import summary
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(device)
# # model = AttentionUNet3D(1, 3)
# # model = model.to(device)
# # summary(model, (1, 16, 144, 144))
# if __name__ == '__main__':
#     torch.cuda.set_device(4)
#     initial_memory = torch.cuda.memory_allocated()
#     print(f"初始显存使用: {initial_memory / (1024 ** 3):.2f} GB")

#     inputs = torch.randn(2,1,192,160,192).to('cuda')
#     model = AttentionUNet3D(1,1).to('cuda')
#     y = model(inputs)
#     memory_used = torch.cuda.memory_allocated() - initial_memory
#     print(f"运行后显存使用: {memory_used / (1024 ** 3):.2f} GB")

import torch
from torch import nn
import torch.nn.functional as F

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

class GridAttentionBlock3D(nn.Module):
    def __init__(self, x_ch, g_ch, sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock3D, self).__init__()

        self.W = nn.Sequential(
            nn.Conv3d(x_ch, x_ch, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, x_ch))
        self.theta = nn.Conv3d(x_ch, x_ch, kernel_size=sub_sample_factor, stride=sub_sample_factor, padding=0, bias=False)
        self.phi = nn.Conv3d(g_ch, x_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(x_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='trilinear', align_corners=True)

        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='trilinear', align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y

class UpsampleAndConcat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpsampleAndConcat, self).__init__()
        self.up_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=True)
        return torch.cat([x, skip], dim=1)

class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class AttentionUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet3D, self).__init__()
        
        # Encoder layers
        self.encoder1 = EncoderConv(n_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = EncoderConv(32, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = EncoderConv(32, 32)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = EncoderConv(32, 64)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # self.encoder5 = EncoderConv(64, 128)
        
        # Attention layers
        self.attention2 = GridAttentionBlock3D(32, 32)
        self.attention3 = GridAttentionBlock3D(32, 32)
        self.attention4 = GridAttentionBlock3D(64, 64)
        
        # Decoder layers
        self.decoder1 = DecoderConv(64 + 64, 64)
        self.up_and_concat1 = UpsampleAndConcat(64, 64)
        
        self.decoder2 = DecoderConv(64 + 32, 32)
        self.up_and_concat2 = UpsampleAndConcat(64, 32)
        
        self.decoder3 = DecoderConv(32 + 32, 12)
        self.up_and_concat3 = UpsampleAndConcat(32, 32)
        
        # self.decoder4 = DecoderConv(32, n_classes)
        
        # Output layer
        self.out_conv = nn.Conv3d(12, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e1_pool = self.pool1(e1)
        e2 = self.encoder2(e1_pool)
        e2_pool = self.pool2(e2)
        e3 = self.encoder3(e2_pool)
        e3_pool = self.pool3(e3)
        e4 = self.encoder4(e3_pool)
        # e4_pool = self.pool4(e4)
        # e5 = self.encoder5(e4_pool)
        
        # Attention
        e2_attention = self.attention2(e2, e2)
        e3_attention = self.attention3(e3, e3)
        e4_attention = self.attention4(e4, e4)
        
        # Decoder
        d4 = e4_attention
        d1 = self.up_and_concat1(d4, e4_attention)
        d1 = self.decoder1(d1)
        
        d2 = self.up_and_concat2(d1, e3_attention)
        d2 = self.decoder2(d2)
        
        d3 = self.up_and_concat3(d2, e2_attention)
        d3 = self.decoder3(d3)
        
        # d4 = self.decoder4(d3)
        d3 = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=True)
        # Upsample d4
        # d4 = F.interpolate(d4, scale_factor=2, mode='trilinear', align_corners=True)

        # Output
        out = self.out_conv(d3)
        return out
    
if __name__ == '__main__':
    torch.cuda.set_device(0)
    initial_memory = torch.cuda.memory_allocated()
    print(f"初始显存使用: {initial_memory / (1024 ** 3):.2f} GB")
    inputs = torch.randn(2,1,192,160,192).to('cuda')
    model = AttentionUNet3D(1,12).to('cuda')
    y = model(inputs)
    print(y.shape)  
    memory_used = torch.cuda.memory_allocated() - initial_memory
    print(f"运行后显存使用: {memory_used / (1024 ** 3):.2f} GB")