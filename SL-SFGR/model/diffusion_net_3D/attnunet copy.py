import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32, 32, 32, 64, 128]):
        super(AttentionUNet3D, self).__init__()
        self.encoder1 = ConvBlock(in_channels, features[0])
        self.encoder2 = ConvBlock(features[0], features[1])
        self.encoder3 = ConvBlock(features[1], features[2])
        self.encoder4 = ConvBlock(features[2], features[3])
        self.encoder5 = ConvBlock(features[3], features[4])
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(features[4], features[4]*2)
        
        self.upconv5 = UpConv(features[4]*2, features[4])
        self.att5 = AttentionBlock(F_g=features[4], F_l=features[4], F_int=features[4]//2)
        self.decoder5 = ConvBlock(features[4]*2, features[4])
        
        self.upconv4 = UpConv(features[4], features[3])
        self.att4 = AttentionBlock(F_g=features[3], F_l=features[3], F_int=features[3]//2)
        self.decoder4 = ConvBlock(features[3]*2, features[3])
        
        self.upconv3 = UpConv(features[3], features[2])
        self.att3 = AttentionBlock(F_g=features[2], F_l=features[2], F_int=features[2]//2)
        self.decoder3 = ConvBlock(features[2]*2, features[2])
        
        self.upconv2 = UpConv(features[2], features[1])
        self.att2 = AttentionBlock(F_g=features[1], F_l=features[1], F_int=features[1]//2)
        self.decoder2 = ConvBlock(features[1]*2, features[1])
        
        self.upconv1 = UpConv(features[1], features[0])
        self.att1 = AttentionBlock(F_g=features[0], F_l=features[0], F_int=features[0]//2)
        self.decoder1 = ConvBlock(features[0]*2, features[0])
        
        self.conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))
        
        bottleneck = self.bottleneck(self.pool(enc5))
        
        dec5 = self.upconv5(bottleneck)
        att5 = self.att5(g=dec5, x=enc5)
        dec5 = self.decoder5(torch.cat((dec5, att5), dim=1))
        
        dec4 = self.upconv4(dec5)
        att4 = self.att4(g=dec4, x=enc4)
        dec4 = self.decoder4(torch.cat((dec4, att4), dim=1))
        
        dec3 = self.upconv3(dec4)
        att3 = self.att3(g=dec3, x=enc3)
        dec3 = self.decoder3(torch.cat((dec3, att3), dim=1))
        
        dec2 = self.upconv2(dec3)
        att2 = self.att2(g=dec2, x=enc2)
        dec2 = self.decoder2(torch.cat((dec2, att2), dim=1))
        
        dec1 = self.upconv1(dec2)
        att1 = self.att1(g=dec1, x=enc1)
        dec1 = self.decoder1(torch.cat((dec1, att1), dim=1))
        
        return self.conv(dec1)

# Example usage:

if __name__ == '__main__':
    torch.cuda.set_device(0)
    initial_memory = torch.cuda.memory_allocated()
    print(f"初始显存使用: {initial_memory / (1024 ** 3):.2f} GB")

    inputs = torch.randn(2,1,192,160,192).to('cuda')
    model = AttentionUNet3D(in_channels=1, out_channels=12).to('cuda')
    y = model(inputs)
    print(y.shape)  
    memory_used = torch.cuda.memory_allocated() - initial_memory
    print(f"运行后显存使用: {memory_used / (1024 ** 3):.2f} GB")