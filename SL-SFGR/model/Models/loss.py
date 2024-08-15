import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import paddle

#动态加权算法
class DynamicWeightAverage(paddle.nn.Layer):
    def __init__(self, losses):
        super(DynamicWeightAverage, self).__init__()
        self.losses = losses
        self.weights = paddle.to_tensor([1.0] * len(losses), stop_gradient=False)
        self.last_losses = [None] * len(losses)

    def forward(self, logits, labels):
        current_losses = [loss(logits, labels).detach() for loss in self.losses]
        
        if any([last_loss is None for last_loss in self.last_losses]):
            self.last_losses = current_losses
            return sum(current_losses)
        
        ratios = [current / last for current, last in zip(current_losses, self.last_losses)]
        exp_ratios = [paddle.exp(ratio) for ratio in ratios]
        
        weights = [len(self.losses) * exp_ratio / sum(exp_ratios) for exp_ratio in exp_ratios]
        self.weights = paddle.to_tensor(weights, stop_gradient=False)
        
        self.last_losses = current_losses
        weighted_loss_sum = sum(w * loss(logits, labels) for w, loss in zip(self.weights, self.losses))
        
        return weighted_loss_sum

class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
        if(self.penalty == "l2"):
            dD = dD * dD
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dD) + torch.mean(dH) + torch.mean(dW)) / 3.0
        return loss

class crossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9, 9), voxel_weights=None):
        super(crossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1], self.kernel[2]])).cuda()


    def forward(self, input, target):
        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0]-1)/2), int((self.kernel[1]-1)/2), int((self.kernel[2]-1)/2))
        T_sum = F.conv3d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv3d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv3d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv3d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv3d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1] * self.kernel[2]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat*T_sum - That*I_sum + That*Ihat*kernelSize
        T_var = TT_sum - 2*That*T_sum + That*That*kernelSize
        I_var = II_sum - 2*Ihat*I_sum + Ihat*Ihat*kernelSize
        cc = cross*cross / (T_var*I_var+1e-5)

        loss = -1.0 * torch.mean(cc)
        return loss
