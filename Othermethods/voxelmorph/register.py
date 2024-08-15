#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
import SimpleITK as sitk
# parse commandline args
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import pystrum.pynd.ndutils as nd
from scipy.spatial import distance
import pystrum.pynd.ndutils as nd
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import subprocess
import glob
import os
import random

def compute_surface_points(binary_image):
    # 提取二值化图像的表面点
    surface_points = np.argwhere(binary_image > 0)
    return surface_points

def compute_assd(surface_points_a, surface_points_b):
    # 计算从A到B的最近距离
    distances_a_to_b = cdist(surface_points_a, surface_points_b).min(axis=1)
    # 计算从B到A的最近距离
    distances_b_to_a = cdist(surface_points_b, surface_points_a).min(axis=1)
    # 计算平均对称表面距离
    assd = (distances_a_to_b.mean() + distances_b_to_a.mean()) / 2.0
    return assd

def compute_assd_2d(binary_image_a, binary_image_b):
    # 选择图像的中间截面
    z_slice = binary_image_a.shape[0] // 2
    
    # 提取截面
    slice_a = binary_image_a[z_slice, :, :]
    slice_b = binary_image_b[z_slice, :, :]
    
    # 计算截面的表面点
    surface_points_a = compute_surface_points(slice_a)
    surface_points_b = compute_surface_points(slice_b)
    
    # 计算平均对称表面距离
    assd = compute_assd(surface_points_a, surface_points_b)
    
    return assd

def dice_val(input, target, num_classes):
    input = torch.tensor(input, dtype=torch.int)
    target = torch.tensor(target, dtype=torch.int)
    smooth = 1.0
    dice_loss = 0.0
        # input = torch.argmax(input,dim=0).int()
    for class_idx in range(1, num_classes):  # 注意跳过背景类别
        input_class = (input==class_idx).int().contiguous().view(-1).to(input.device)
        target_class = (target==class_idx).int().contiguous().view(-1).to(input.device)

        intersection = torch.sum(input_class * target_class)
        cardinality = torch.sum(input_class) + torch.sum(target_class)

        dice_coeff = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_loss += 1.0 - dice_coeff

    dice_loss /= num_classes - 1  # 对所有类别的Dice Loss取平均

    return dice_loss


class Jacobians:

    def Get_Jac(displacement):
        '''
        the expected input: displacement of shape(batch, H, W, D, channel),
        obtained in TensorFlow.
        '''
        displacement=np.transpose(displacement,(0,2,3,4,1))
        D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])
    
        D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,1]*D_z[...,1])
        D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
        D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
        
        D = D1 - D2 + D3

        negative_elements=D[D<0]
        persent = len(negative_elements)/np.size(D)
        std_deviation=np.std(D)
        
        return persent,std_deviation


class JAC:
    @staticmethod
    def calculate_jacobian_metrics(disp):
        """
        Calculate Jacobian related regularity metrics.
        Args:
            disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field
        Returns:
            folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
            mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
        """
        negative_det_J = []
        mag_grad_det_J = []
        std_log_det_J = []
        for n in range(disp.shape[0]):
            disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
            jac_det_n = JAC.jacobian_det(disp_n)
            negative_det_J += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
            mag_grad_det_J += [np.abs(np.gradient(jac_det_n)).mean()]
            std_log_det_J += [np.log(jac_det_n.clip(1e-9, 1e9)).std()]
        return {
            'negative_det_J': np.mean(negative_det_J),
            'mag_grad_det_J': np.mean(mag_grad_det_J),
            'std_log_det_J': np.mean(std_log_det_J)
        }

    @staticmethod
    def jacobian_det(disp):
        """
        Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)
        Args:
            disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field
        Returns:
            jac_det: (numpy.ndarray, shape (*sizes)) Point-wise Jacobian determinant
        """
        disp_img = sitk.GetImageFromArray(disp.astype('float32'), isVector=True)
        jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
        jac_det = sitk.GetArrayFromImage(jac_det_img)
        return jac_det

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """
    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    # compute gradients
    J = np.gradient(disp + grid)
    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]
        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        sums = Jdet0 - Jdet1 + Jdet2
        return sums, np.sum(sums<0)/np.size(sums)
    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        sums = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

        return sums, np.sum(sums<0)/np.size(sums)
def gradient_magnitude(jac_det):
    # Compute the gradient of the Jacobian determinant
    grad = np.gradient(jac_det)
    # Compute the magnitude of the gradient
    grad_magnitude = np.sqrt(np.sum(np.square(grad), axis=0))
    return np.linalg.norm(grad_magnitude)
    # return grad_magnitude


def hausdorff_distance(seg1, seg2):
    """
    Compute the Hausdorff Distance between two binary segmentations.

    Parameters:
        seg1: np.ndarray - binary segmentation of the first object
        seg2: np.ndarray - binary segmentation of the second object

    Returns:
        hd: float - Hausdorff Distance
        hd95: float - 95th percentile Hausdorff Distance
    """
    # Extract the boundary points
    seg1_points = np.argwhere(seg1)
    seg2 = seg2.squeeze().cpu().numpy()
    seg2_points = np.argwhere(seg2)

    # Compute all distances from points in seg1 to seg2 and vice versa
    dists_1_to_2 = distance.cdist(seg1_points, seg2_points, 'euclidean')
    dists_2_to_1 = distance.cdist(seg2_points, seg1_points, 'euclidean')

    # Hausdorff Distance
    hd = max(dists_1_to_2.max(axis=1).min(), dists_2_to_1.max(axis=1).min())

    # 95th percentile Hausdorff Distance
    hd95 = np.percentile(np.hstack((dists_1_to_2.min(axis=1), dists_2_to_1.min(axis=1))), 95)

    return hd, hd95

def IOU_3D(y_pred, y, num_classes=12):
    # y：真实值，y_pred预测值
    epoch_iou = []

    # 转换 y_pred 到与 y 相同的设备
    y_pred = torch.from_numpy(y_pred).to(y.device)
    
    for i in range(1,num_classes):
        # 构建二值化掩码
        temp_y = (y == i).float()
        temp_y_pred = (y_pred == i).float()

        intersection = torch.logical_and(temp_y, temp_y_pred)
        union = torch.logical_or(temp_y, temp_y_pred)

        # 计算 IoU
        batch_iou = torch.true_divide(torch.sum(intersection), torch.sum(union))
        epoch_iou.append(batch_iou)
    
    # 返回平均 IoU
    return torch.mean(torch.tensor(epoch_iou))

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
        src = torch.tensor(src,dtype=torch.float32)
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
parser = argparse.ArgumentParser()
parser.add_argument('--moving', help='moving image (source) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30036_MR_d1199.nii.gz')
parser.add_argument('--moving_label', help='moving image (source) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30036_MR_d1199.nii.gz')


parser.add_argument('--fixed', help='fixed image (target) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30009_MR_d3534.nii.gz')
parser.add_argument('--fixed_label', help='fixed image (target) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30009_MR_d3534.nii.gz')

parser.add_argument('--moved', help='warped image output filename',default='')
parser.add_argument('--model', help='pytorch model for nonlinear registration',default='/root/autodl-tmp/LungRe/voxelmorph/models/bird_MRI_NCC/1500.pt')
parser.add_argument('--warp', help='output warp deformation filename',default='')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used',default='0')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# 获取文件路径列表
directory_path = "/root/autodl-tmp/OASIS3_Dataset/test/"
nii_gz_files = glob.glob(os.path.join(directory_path, "*.nii.gz"))
files = [f for f in nii_gz_files if not f.endswith("_labels.nii.gz")]
label_files = glob.glob(os.path.join(directory_path, "*_labels.nii.gz"))
model_path = '/home/vrdoc/GF/lung_registration/voxelmorph/models/MRI_NCC_good_model/1500.pt'
sorted(files)
sorted(label_files)
# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load moving and fixed images
add_feat_axis = not args.multichannel
    # load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()
with open('/root/autodl-tmp/results/vm_bird_result.txt','a+') as f:
    for i in range(len(files)):
        index = random.randint(0,len(files)-1)
        while(index==i):
            index = random.randint(0,len(files)-1)
        args.moving = files[i]
        args.fixed =files[index]
        args.moving_label = label_files[i]
        args.fixed_label = label_files[i]
        moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed, fixed_affine = vxm.py.utils.load_volfile(
            args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        moving_label =  vxm.py.utils.load_volfile(args.moving_label, add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_label =  vxm.py.utils.load_volfile(args.fixed_label, add_batch_axis=True, add_feat_axis=add_feat_axis)
        # set up tensors and permute
        input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
        input_moving = (input_moving-input_moving.min())/(input_moving.max()-input_moving.min())
        input_fixed = (input_fixed-input_fixed.min())/(input_fixed.max()-input_fixed.min())
        STN = SpatialTransformer((160,192,192)).to('cuda:0')
        # predict
        moved, warp = model(input_moving, input_fixed, registration=True)
        warped_mask = STN(torch.from_numpy(moving_label).to(warp.device).permute(0,4,1,2,3),warp)
        moved = moved.detach().cpu().numpy().squeeze()
        warped_mask = warped_mask.detach().cpu().numpy().squeeze()
        warp = warp.detach().cpu().numpy()
        fixed_label = fixed_label.squeeze()

        ground_truth_image = (fixed_label > 0).astype(np.uint8)
        prediction_image = (warped_mask > 0).astype(np.uint8)

        dice = dice_val(warped_mask,fixed_label,12)
        Jac = JAC().calculate_jacobian_metrics(warp)
        ASSDs = compute_assd_2d(ground_truth_image, prediction_image)
        # iou =(IOU_3D(seg_out_f,test_data['FL'])+IOU_3D(seg_out_m,test_data['ML']))/2.0
        print(dice,Jac,ASSDs)
        f.writelines("Dica:{},negative_det_J:{},mag_grad_det_J:{},'std_log_det_J':{},ASSDS:{}\n".format(dice,Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs))


