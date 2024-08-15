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
def dice_val(input, target, num_classes):
    input = torch.tensor(input, dtype=torch.int)
    target = torch.tensor(target, dtype=torch.int)
    smooth = 1.0
    dice_loss = 0.0
        # input = torch.argmax(input,dim=0).int()
    for class_idx in range(1, num_classes):  # 注意跳过背景类别
        input_class = (input==class_idx).int().contiguous().view(-1)
        target_class = (target==class_idx).int().contiguous().view(-1)

        intersection = torch.sum(input_class * target_class)
        cardinality = torch.sum(input_class) + torch.sum(target_class)

        dice_coeff = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_loss += 1.0 - dice_coeff

    dice_loss /= num_classes - 1  # 对所有类别的Dice Loss取平均

    return dice_loss


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
        return Jdet0 - Jdet1 + Jdet2
    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def gradient_magnitude(jac_det):
    # Compute the gradient of the Jacobian determinant
    grad = np.gradient(jac_det)
    # Compute the magnitude of the gradient
    grad_magnitude = np.sqrt(np.sum(np.square(grad), axis=0))
    
    return grad_magnitude


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
    seg2_points = np.argwhere(seg2)

    # Compute all distances from points in seg1 to seg2 and vice versa
    dists_1_to_2 = distance.cdist(seg1_points, seg2_points, 'euclidean')
    dists_2_to_1 = distance.cdist(seg2_points, seg1_points, 'euclidean')

    # Hausdorff Distance
    hd = max(dists_1_to_2.max(axis=1).min(), dists_2_to_1.max(axis=1).min())

    # 95th percentile Hausdorff Distance
    hd95 = np.percentile(np.hstack((dists_1_to_2.min(axis=1), dists_2_to_1.min(axis=1))), 95)

    return hd, hd95

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
parser = argparse.ArgumentParser()
parser.add_argument('--moving', help='moving image (source) filename',default='/root/autodl-tmp/OASIS3_Dataset/test/OAS30028_MR_d1260.nii.gz')
# parser.add_argument('--moving', help='moving image (source) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30036_MR_d1199.nii.gz')
# parser.add_argument('--moving_label', help='moving image (source) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30036_MR_d1199.nii.gz')


parser.add_argument('--fixed', help='fixed image (target) filename',default='/root/autodl-tmp/OASIS3_Dataset/test/OAS30004_MR_d3457.nii.gz')
# parser.add_argument('--fixed_label', help='fixed image (target) filename',default='/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/test/OAS30009_MR_d3534.nii.gz')

parser.add_argument('--moved', help='warped image output filename',default='OAS30028_MR_d1260_to_OAS30004_MR_d3457_warped.nii.gz')
parser.add_argument('--model', help='pytorch model for nonlinear registration',default='/root/autodl-tmp/LungRe/voxelmorph/models/bird_MRI_NCC/1500.pt')
parser.add_argument('--warp', help='output warp deformation filename',default='OAS30028_MR_d1260_to_OAS30004_MR_d3457_flow.nii.gz')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used',default='0')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# moving_label =  vxm.py.utils.load_volfile(args.moving_label, add_batch_axis=True, add_feat_axis=add_feat_axis)
# fixed_label =  vxm.py.utils.load_volfile(args.fixed_label, add_batch_axis=True, add_feat_axis=add_feat_axis)


# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# set up tensors and permute
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

input_moving = (input_moving-input_moving.min())/(input_moving.max()-input_moving.min())
input_fixed = (input_fixed-input_fixed.min())/(input_fixed.max()-input_fixed.min())
STN = SpatialTransformer((192,192,160))
# predict
moved, warp = model(input_moving, input_fixed, registration=True)
# warped_mask = STN(moving_label,warp)

# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze().transpose(1,2,3,0)
    print(warp.shape)
    sitk.WriteImage(sitk.GetImageFromArray(warp),args.warp)
    # vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)
# moved = moved.detach().cpu().numpy().squeeze()
# warped_mask = warped_mask.detach().cpu().numpy().squeeze()
# warp = warp.detach().cpu().numpy().squeeze().transpose(1,2,3,0)

# jac = jacobian_determinant(warp)
# print(dice_val(warped_mask,fixed_label,12),jac,gradient_magnitude(jac),hausdorff_distance(warped_mask,fixed_label))

