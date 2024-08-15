import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
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


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self,y_pred,y_true):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1.0-torch.mean(cc)

class crossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9, 9), voxel_weights=None):
        super(crossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1], self.kernel[2]]))


    def forward(self, input, target):
        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0]-1)/2), int((self.kernel[1]-1)/2), int((self.kernel[2]-1)/2))
        T_sum = F.conv3d(target, self.filt.to(input.device), stride=1, padding=pad)
        I_sum = F.conv3d(input, self.filt.to(input.device), stride=1, padding=pad)
        TT_sum = F.conv3d(TT, self.filt.to(input.device), stride=1, padding=pad)
        II_sum = F.conv3d(II, self.filt.to(input.device), stride=1, padding=pad)
        IT_sum = F.conv3d(IT, self.filt.to(input.device), stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1] * self.kernel[2]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat*T_sum - That*I_sum + That*Ihat*kernelSize
        T_var = TT_sum - 2*That*T_sum + That*That*kernelSize
        I_var = II_sum - 2*Ihat*I_sum + Ihat*Ihat*kernelSize
        cc = cross*cross / (T_var*I_var+1e-5)

        loss = 1.0 - torch.mean(cc)
        return loss

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):
        smooth = 0.01

        input = input.view(-1)
        target = target.view(-1)

        inter = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target) + smooth

        score = 2.0 * inter / union
        score = 1.0 - score

        return score

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        smooth = 1.0
        dice_loss = 0.0
        # input = torch.argmax(input,dim=0).int()
        for class_idx in range(1, self.num_classes):  # 注意跳过背景类别
            input_class = (input==class_idx).int().contiguous().view(-1)
            target_class = (target==class_idx).int().contiguous().view(-1)

            intersection = torch.sum(input_class * target_class)
            cardinality = torch.sum(input_class) + torch.sum(target_class)

            dice_coeff = (2.0 * intersection + smooth) / (cardinality + smooth)
            dice_loss += 1.0 - dice_coeff

        dice_loss /= self.num_classes - 1  # 对所有类别的Dice Loss取平均
        dice_loss = 1-dice_loss

        return dice_loss
    def encode_one_hot(self,label_map, num_classes):
        """
        Encode label map into one-hot representation.

        Args:
        - label_map: The label map with shape [depth, height, width].
        - num_classes: The total number of classes.

        Returns:
        - one_hot_map: The one-hot representation of the label map with shape [depth, height, width, num_classes].
        """
        one_hot_map = np.zeros(label_map.shape + (num_classes,), dtype=np.float32)
        for class_idx in range(num_classes):
            one_hot_map[..., class_idx] = (label_map == class_idx).astype(np.float32)
        return one_hot_map
    def decode_one_hot(self,one_hot_map):
        """
        Decode one-hot representation into label map.

        Args:
        - one_hot_map: The one-hot representation of the label map with shape [depth, height, width, num_classes].

        Returns:
        - label_map: The decoded label map with shape [depth, height, width].
        """
        label_map = np.argmax(one_hot_map, axis=-1)
        return label_map