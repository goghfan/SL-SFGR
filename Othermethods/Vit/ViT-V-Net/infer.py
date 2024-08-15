import glob
from torch.utils.tensorboard import SummaryWriter
import logging
import os, losses, utils
import shutil
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
import pystrum.pynd.ndutils as nd
# from ignite.contrib.handlers import ProgressBar
from torchsummary import summary
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
import data as datass
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import SimpleITK as sitk

device = 'cuda:0'


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

def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], linewidth=0.8, **kwargs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))


def set_device(device, x):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None and not isinstance(item, list):
                x[key] = item.to(device)
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.to(device)
    else:
        x = x.to(device)
    return x

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


def main():
    #等待补充test路径
    test_dir = '/root/autodl-tmp/'
    model_idx = -1
    model_folder = '/root/autodl-tmp/BIBM2024/ViT-V-Net_for_3D_Image_Registration_Pytorch/result_transmorph'
    model_dir = '/root/autodl-tmp/BIBM2024/ViT-V-Net_for_3D_Image_Registration_Pytorch/result/'
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    # dict = utils.process_label()
    if os.path.exists(model_folder[:-1]+'.csv'):
        os.remove(model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], model_folder[:-1])
    line = ''
    for i in range(1,12):
        line = line + ',' + str(i)
    csv_writter(line, model_folder[:-1])
    model = models.ViTVNet(config_vit, img_size=(192,192,160))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx],map_location='cuda:0')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((192,192,160), 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    phase='test'
    batchSize = 1
    test_set = datass.create_dataset(test_dir,trainortest=phase)
    test_loader = datass.create_dataloader(test_set, batchSize, phase)    
    
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    
    with open('vit_result.txt','a+') as f:
        with torch.no_grad():
            stdy_idx = 0
            for data in test_loader:
                model.eval()
                data = set_device(device,data)
                x = data['M']
                y = data['F']
                x_seg = data['ML']
                y_seg = data['FL']

                x_in = torch.cat((x,y),dim=1)
                x_def, flow = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                #jac_det = utils.jacobian_determinant(flow.detach().cpu().numpy()[0, :, :, :, :])
                line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
                line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
                csv_writter(line, model_folder[:-1])
                #eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

                dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 12)
                dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 12)
                ground_truth_image = (data['ML'].squeeze().cpu().numpy() > 0).astype(np.uint8)
                prediction_image = (def_out.squeeze().cpu().numpy() > 0).astype(np.uint8)
                ASSDs = compute_assd_2d(ground_truth_image,prediction_image)
                Jac = JAC().calculate_jacobian_metrics(flow.cpu().detach().numpy())
                print('Trans diff: {:.4f}, Raw diff: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
                f.writelines('Trans diff: {:.4f}, Raw diff: {:.4f},negative_det_J:{}, mag_grad_det_J:{}, std_log_det_J:{}, ASSDS:{}\n'.format(dsc_trans.item(), dsc_raw.item(),Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs))

                eval_dsc_def.update(dsc_trans.item(), x.size(0))
                eval_dsc_raw.update(dsc_raw.item(), x.size(0))
                stdy_idx += 1

                # flip moving and fixed images
                y_in = torch.cat((y, x), dim=1)
                y_def, flow = model(y_in)
                def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
                tar = x.detach().cpu().numpy()[0, 0, :, :, :]

                #jac_det = utils.jacobian_determinant(flow.detach().cpu().numpy()[0, :, :, :, :])
                line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
                line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
                out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
                #print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
                csv_writter(line,  model_folder[:-1])
                #eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

                dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 12)
                dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 12)

                ground_truth_image = (data['ML'].squeeze().cpu().numpy() > 0).astype(np.uint8)
                prediction_image = (def_out.squeeze().cpu().numpy() > 0).astype(np.uint8)
                ASSDs = compute_assd_2d(ground_truth_image,prediction_image)
                Jac = JAC().calculate_jacobian_metrics(flow.cpu().detach().numpy())

                print('Trans diff: {:.4f}, Raw diff: {:.4f},negative_det_J:{}, mag_grad_det_J:{}, std_log_det_J:{}, ASSDS:{}\n'.format(dsc_trans.item(), dsc_raw.item(),Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs))
                f.writelines('Trans diff: {:.4f}, Raw diff: {:.4f},negative_det_J:{}, mag_grad_det_J:{}, std_log_det_J:{}, ASSDS:{}\n'.format(dsc_trans.item(), dsc_raw.item(),Jac['negative_det_J'],Jac['mag_grad_det_J'],Jac['std_log_det_J'],ASSDs))
                eval_dsc_def.update(dsc_trans.item(), x.size(0))
                eval_dsc_raw.update(dsc_raw.item(), x.size(0))
                stdy_idx += 1

            print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                            eval_dsc_def.std,
                                                                                            eval_dsc_raw.avg,
                                                                                            eval_dsc_raw.std))
            print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
            f.writelines('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                            eval_dsc_def.std,
                                                                                            eval_dsc_raw.avg,
                                                                                            eval_dsc_raw.std))
            f.writelines('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()