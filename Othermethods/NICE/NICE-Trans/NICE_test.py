# py imports
import os
import sys
import glob
import time
import numpy as np
import torch
import scipy.ndimage
from argparse import ArgumentParser
import SimpleITK as sitk
# project imports
import networks
import datagenerators
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import pystrum.pynd.ndutils as nd


def Dice(vol1, vol2, labels=[0,1,2,3,4,5,6,7,8,9,10,11], nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem)
    else:
        return (dicem, labels)


def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)/np.size(Ja_value)


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

def test(test_dir,
         test_pairs,
         device, 
         load_model):
    
    # preparation
    # test_pairs = np.load(test_dir+test_pairs, allow_pickle=True)

    # device handling
    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
    
    # prepare model
    model = networks.NICE_Trans()
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # transfer model
    SpatialTransformer = networks.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval()
    
    AffineTransformer = networks.AffineTransformer_block(mode='nearest')
    AffineTransformer.to(device)
    AffineTransformer.eval()
    
    # testing loop
    Dice_result = [] 
    NJD_result = []
    Affine_result = []
    Runtime_result = []
    with open('/root/autodl-tmp/BIBM2024/NICE/Registration-NICE-Trans/NICE-Trans/NICE_result.txt','a+') as f:
        moving = sitk.ReadImage('/root/autodl-tmp/OASIS3_Dataset/test/OAS30028_MR_d1260.nii.gz')
        moving_label = sitk.ReadImage('/root/autodl-tmp/OASIS3_Dataset/test/OAS30028_MR_d1260_labels.nii.gz')
        fixed = sitk.ReadImage('/root/autodl-tmp/OASIS3_Dataset/test/OAS30004_MR_d3457.nii.gz')
        fixed_label = sitk.ReadImage('/root/autodl-tmp/OASIS3_Dataset/test/OAS30004_MR_d3457_labels.nii.gz')

        moving = sitk.GetArrayFromImage(moving)
        moving_label = sitk.GetArrayFromImage(moving_label)
        fixed = sitk.GetArrayFromImage(fixed)
        fixed_label = sitk.GetArrayFromImage(fixed_label)

        fixed = (fixed - fixed.min())/(fixed.max()-fixed.min())
        moving = (moving - moving.min())/(moving.max()-moving.min())

        fixed = np.expand_dims(fixed, axis=0)
        moving = np.expand_dims(moving, axis=0)
        fixed = np.expand_dims(fixed, axis=0)
        moving = np.expand_dims(moving, axis=0)
            # 移动到GPU
        fixed = torch.from_numpy(fixed).float()
        moving = torch.from_numpy(moving).float()

        fixed_label = np.expand_dims(fixed_label, axis=0)
        moving_label = np.expand_dims(moving_label, axis=0)

        fixed_label = torch.from_numpy(fixed_label).int()
        fixed_label = fixed_label.type(torch.LongTensor) 
        moving_label = torch.from_numpy(moving_label).int()
        moving_label = moving_label.type(torch.LongTensor) 

        test_data = {'M':moving,'F':fixed,'ML':moving_label,'FL':fixed_label}
        dataname=str('OAS30028_MR_d1260.nii.gz'+'_to_'+'OAS30004_MR_d3457.nii.gz')

        with torch.no_grad():
            pred = model(fixed.permute(0,1,4,2,3).to('cuda:0'), moving.permute(0,1,4,2,3).to('cuda:0'))
            warped=pred[0].permute(0,1,4,3,2)
            final_flow=pred[1].permute(0,1,4,3,2)
            final_flow = final_flow.squeeze().cpu().numpy()
            warped = warped.squeeze().cpu().numpy()

            final_flow = np.transpose(final_flow, (1, 2, 3, 0))

            sample_img = sitk.ReadImage('/root/autodl-tmp/OASIS3_Dataset/test/OAS30004_MR_d3457.nii.gz')
            origin = sample_img.GetOrigin()
            direction = sample_img.GetDirection()
            spacing = sample_img.GetSpacing()

            final_flow = sitk.GetImageFromArray(final_flow)
            warped = sitk.GetImageFromArray(warped)

            final_flow.SetDirection(direction)
            final_flow.SetOrigin(origin)
            final_flow.SetSpacing(spacing)

            warped.SetDirection(direction)
            warped.SetOrigin(origin)
            warped.SetSpacing(spacing)

            sitk.WriteImage( final_flow,'/root/autodl-tmp/BIBM2024/NICE/Registration-NICE-Trans/NICE-Trans/'+'OAS30028_MR_d1260_to_OAS30004_MR_d3457'+'_flow.nii.gz')
            sitk.WriteImage( warped,'/root/autodl-tmp/BIBM2024/NICE/Registration-NICE-Trans/NICE-Trans/'+'OAS30028_MR_d1260_to_OAS30004_MR_d3457'+'_warped.nii.gz')


if __name__ == "__main__":
    parser = ArgumentParser()
    #test目录等待补充ing
    parser.add_argument("--test_dir", type=str,
                        dest="test_dir", default='/root/autodl-tmp/OASIS3_Dataset/test/',
                        help="folder with testing data")
    #注意数据集中需要包含test_pairs.npy
    parser.add_argument("--test_pairs", type=str,
                        dest="test_pairs", default='image_pairs.npy',
                        help="testing pairs(.npy)")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    #模型的路径
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='/root/autodl-tmp/BIBM2024/NICE/Registration-NICE-Trans/NICE-Trans/models/301.pt',
                        help="load model file to initialize with")
    args = parser.parse_args()
    test(**vars(args))
