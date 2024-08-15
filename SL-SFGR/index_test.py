import torch
import data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
import torch.nn as nn
import pystrum.pynd.ndutils as nd
from scipy.spatial import distance
from scipy.spatial.distance import cdist


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
def main_test():
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/root/autodl-tmp/LungRe/config/diffuseMorph_test_3D.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            batchSize = opt['datasets']['train']['batch_size']
            train_set = data.create_dataset(opt['datasets']['train']['dataroot'],trainortest=phase)
            train_loader = data.create_dataloader(train_set, batchSize, phase)
            # if torch.cuda.is_available() and torch.cuda.device_count()>1:
            #     train_loader = torch.nn.DataParallel(train_loader)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            batchSize = 1
            test_set = data.create_dataset(opt['datasets']['test']['dataroot'],trainortest=phase)
            test_loader = data.create_dataloader(test_set, batchSize, phase)
    logger.info('Initial dataset Finished')

    # model
    diffusion = Model.create_model(opt)

    logger.info('Initial Model Finished')


    #Test Model
    registTime = []
    logger.info('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path,exist_ok=True)
    with open("/root/autodl-tmp/LungRe/nodice_result.txt",'a+') as f:
        for istep,test_data in enumerate(test_loader):
            idx+=1
            dataname=str(test_data['name_m'][0][-24:-7])+'_to_'+str(test_data['name_f'][0][-24:-7])
            diffusion.feed_data(test_data)

            final_flow,warped,warped_mask,seg_out_m,seg_out_f = diffusion.test_registration()

            final_flow = final_flow.squeeze().cpu().numpy()
            warped = warped.squeeze().cpu().numpy()
            warped_mask = warped_mask.int().squeeze().cpu().numpy()
            seg_out_m = seg_out_m.squeeze().cpu().numpy()
            seg_out_f = seg_out_f.squeeze().cpu().numpy()

            seg_out_f = np.argmax(seg_out_f,axis=0).astype(np.float32)
            seg_out_m = np.argmax(seg_out_m,axis=0).astype(np.float32)

            final_flow =  np.expand_dims(final_flow, axis=0)

            # jacb,jac = jacobian_determinant(final_flow)

            # print(dice_val(warped_mask,test_data['FL'],12),jac,gradient_magnitude(jac),hausdorff_distance(warped_mask,test_data['FL']))
            ground_truth_image = (test_data['FL'].squeeze().cpu().numpy() > 0).astype(np.uint8)
            prediction_image = (warped_mask > 0).astype(np.uint8)

            # surface_points_gt = extract_surface_points(ground_truth_image)
            # surface_points_pred = extract_surface_points(prediction_image)
            dice = dice_val(warped_mask,test_data['FL'],12)
            Jac = JAC().calculate_jacobian_metrics(final_flow)
            ASSDs = compute_assd_2d(ground_truth_image, prediction_image)
            iou =(IOU_3D(seg_out_f,test_data['FL'])+IOU_3D(seg_out_m,test_data['ML']))/2.0
            print(dice,Jac,ASSDs,iou)
            f.writelines("Dica:{},negative_det_J:{},mag_grad_det_J:{},'std_log_det_J':{},ASSDS:{},IOU{}\n".format(dice,Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs,iou))

def vm_test():
    import subprocess
    import glob
    import os
    import random
# 获取文件路径列表
    directory_path = "/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/validation/"
    nii_gz_files = glob.glob(os.path.join(directory_path, "*.nii.gz"))
    files = [f for f in nii_gz_files if not f.endswith("_labels.nii.gz")]
    label_files = glob.glob(os.path.join(directory_path, "*_labels.nii.gz"))
    model_path = '/home/vrdoc/GF/lung_registration/voxelmorph/models/MRI_NCC_good_model/1500.pt'
    sorted(files)
    sorted(label_files)
    parameters_list ={
        "moving": "moving1.nii",
        "fixed": "fixed1.nii",
        "moved": "moved1.nii",
        "model": model_path,
        "warp": "warp1.nii",
        "gpu": "0"
    }
    for i in range(len(files)):
        index = random.randint(0,len(files))
        while(index==i):
            index = random.randint(0,len(files))
        parameters_list["moving"] = files[i]
        parameters_list["fixed"] = files[index]
        parameters_list['moved'] = files[i][:-7]+"moved_to"+files[index]
        parameters_list['warp'] = files[i][:-7]+"_to_"+files[index][:-7]+"warped.nii.gz"
        parameters_list['moving_label'] = label_files[i]
        parameters_list['fixed_label'] = label_files[i]
        cmd = [
            "python", "register.py",
            "--moving", parameters_list["moving"],
            "--moving_label",parameters_list['moving_label'],
            "--fixed", parameters_list["fixed"],
            "--fixed_label", parameters_list["fixed_label"],
            "--moved", parameters_list["moved"],
            "--model", parameters_list["model"],
            "--warp", parameters_list["warp"],
            "-g", parameters_list["gpu"]]
        # 执行命令
        subprocess.run(cmd)

def pivit_test():
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/root/autodl-tmp/LungRe/config/diffuseMorph_test_3D.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            batchSize = opt['datasets']['train']['batch_size']
            train_set = data.create_dataset(opt['datasets']['train']['dataroot'],trainortest=phase)
            train_loader = data.create_dataloader(train_set, batchSize, phase)
            # if torch.cuda.is_available() and torch.cuda.device_count()>1:
            #     train_loader = torch.nn.DataParallel(train_loader)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            batchSize = 1
            test_set = data.create_dataset(opt['datasets']['test']['dataroot'],trainortest=phase)
            test_loader = data.create_dataloader(test_set, batchSize, phase)
    logger.info('Initial dataset Finished')

    # model
    diffusion = Model.create_model(opt)

    logger.info('Initial Model Finished')


    #Test Model
    registTime = []
    logger.info('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path,exist_ok=True)
    with open("/root/autodl-tmp/results/pivit.result.txt",'a+') as f:
        for istep,test_data in enumerate(test_loader):
            idx+=1
            dataname=str(test_data['name_m'][0][-24:-7])+'_to_'+str(test_data['name_f'][0][-24:-7])
            diffusion.feed_data(test_data)

            warped,warped_mask,final_flow = diffusion.test_registration_origin()

            final_flow = final_flow.squeeze().cpu().numpy()
            warped = warped.squeeze().cpu().numpy()
            warped_mask = warped_mask.int().squeeze().cpu().numpy()
            # seg_out_m = seg_out_m.squeeze().cpu().numpy()
            # seg_out_f = seg_out_f.squeeze().cpu().numpy()

            # seg_out_f = np.argmax(seg_out_f,axis=0).astype(np.float32)
            # seg_out_m = np.argmax(seg_out_m,axis=0).astype(np.float32)

            final_flow =  np.expand_dims(final_flow, axis=0)

            # jacb,jac = jacobian_determinant(final_flow)

            # print(dice_val(warped_mask,test_data['FL'],12),jac,gradient_magnitude(jac),hausdorff_distance(warped_mask,test_data['FL']))
            ground_truth_image = (test_data['FL'].squeeze().cpu().numpy() > 0).astype(np.uint8)
            prediction_image = (warped_mask > 0).astype(np.uint8)

            # surface_points_gt = extract_surface_points(ground_truth_image)
            # surface_points_pred = extract_surface_points(prediction_image)
            dice = dice_val(warped_mask,test_data['FL'],12)
            Jac = JAC().calculate_jacobian_metrics(final_flow)
            ASSDs = compute_assd_2d(ground_truth_image, prediction_image)
            # iou =(IOU_3D(seg_out_f,test_data['FL'])+IOU_3D(seg_out_m,test_data['ML']))/2.0
            print(dice,Jac,ASSDs)
            f.writelines("Dice:{},negative_det_J:{},mag_grad_det_J:{},'std_log_det_J':{},ASSDS:{}\n".format(dice,Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs))

            # f.writelines("Dica:{},negative_det_J:{},mag_grad_det_J:{},'std_log_det_J':{},ASSDS:{},IOU{}\n".format(dice,Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs,iou))




if __name__ == "__main__":
    pivit_test()