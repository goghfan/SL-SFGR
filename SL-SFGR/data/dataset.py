import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import SimpleITK as sitk
import random


class CT_Datasets(Dataset):
    def __init__(self, dataroot=r'data/', trainortest='Train'):
        self.trainortest = trainortest
        self.fixed_path = glob.glob(os.path.join(
            dataroot, trainortest+'/Fixed/', "*.mha"))
        self.moving_path = glob.glob(os.path.join(
            dataroot, trainortest+'/Moving/', "*.mha"))
        self.data_len = len(self.moving_path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        indexF = random.randint(0, len(self.fixed_path)-1)
        # 读取数据
        Moving_Case_num = int(self.moving_path[index][self.moving_path[index].find(
            'Case')+4:self.moving_path[index].find('_T')])
        # fixed_img = sitk.ReadImage('data/'+ 'train' +'/Fixed/'+'Case{}_T0.mha'.format(Moving_Case_num))
        fixed_img = sitk.ReadImage(self.fixed_path[indexF])
        fixed_img = sitk.GetArrayFromImage(fixed_img)
        moving_img = sitk.ReadImage(self.moving_path[index])
        moving_img = sitk.GetArrayFromImage(moving_img)
        fixed_img = (fixed_img - fixed_img.min()) / \
            (fixed_img.max()-fixed_img.min())
        moving_img = (moving_img - moving_img.min()) / \
            (moving_img.max()-moving_img.min())
        # 扩展fixed_img和moving_img为[B,C,D,H,W]
        fixed_img = np.expand_dims(fixed_img, axis=0)
        moving_img = np.expand_dims(moving_img, axis=0)
        # 移动到GPU
        fixed_img = torch.from_numpy(fixed_img).float()
        moving_img = torch.from_numpy(moving_img).float()
        # if self.trainortest == 'train':
        #     return {'F':fixed_img,'M':moving_img}
        # else:
        # 根据文件路径寻找对应label文件路径 fixed_img = 'data\Train\Fixed\Case1_T0.mha'
        Fixed_Case_num = Moving_Case_num
        # Fixed_T_num = int(self.fixed_path[indexF][self.fixed_path[indexF].find('_T')+2:self.fixed_path[indexF].find('.mha')])
        Moving_T_num = int(self.moving_path[index][self.moving_path[index].find(
            '_T')+2:self.moving_path[index].find('.mha')])
        F_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(
            'data/maskdata/C{}T00_lungmask.mha'.format(Fixed_Case_num)))
        M_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(
            'data/maskdata/C{}T{}_lungmask.mha'.format(Moving_Case_num, Moving_T_num)))
        F_mask_img = np.expand_dims(F_mask_img, axis=0)
        M_mask_img = np.expand_dims(M_mask_img, axis=0)
        F_mask_img = torch.from_numpy(F_mask_img).float()
        M_mask_img = torch.from_numpy(M_mask_img).float()

        F_vessel_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(
            'data/vessel_maskdata/C{}T00_vesselmask.mha'.format(Fixed_Case_num)))
        M_vessel_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(
            'data/vessel_maskdata/C{}T{}_vesselmask.mha'.format(Moving_Case_num, Moving_T_num)))
        F_vessel_mask_img = np.expand_dims(F_vessel_mask_img, axis=0)
        M_vessel_mask_img = np.expand_dims(M_vessel_mask_img, axis=0)
        F_vessel_mask_img = torch.from_numpy(F_vessel_mask_img).float()
        M_vessel_mask_img = torch.from_numpy(M_vessel_mask_img).float()

        return {'F': fixed_img, 'M': moving_img, 'FL': F_mask_img, 'ML': M_mask_img, 'FV': F_vessel_mask_img, 'MV': M_vessel_mask_img, 'name_f': 'C{}T00'.format(Fixed_Case_num), 'name_m': 'C{}T{}'.format(Moving_Case_num, Moving_T_num)}


class MRI_Datasets(Dataset):
    def __init__(self, dataroot=r'data/', trainortest='Train'):
        self.trainortest = trainortest
        self.image_path = glob.glob(os.path.join(dataroot, "OASIS3_Dataset/"+trainortest, "*.nii.gz"))
        self.label_path = glob.glob(os.path.join(dataroot, "OASIS3_Dataset/"+trainortest, "*_labels.nii.gz"))
        self.image_path = [i for i in self.image_path if i not in self.label_path]
        self.label_path.sort()
        self.image_path.sort()
        self.data_len = len(self.image_path)

    def __len__(self):
        return self.data_len

    def encode_one_hot(self,label_map, num_classes):
        """
        Encode label map into one-hot representation.

        Args:
        - label_map: The label map with shape [depth, height, width].
        - num_classes: The total number of classes.

        Returns:
        - one_hot_map: The one-hot representation of the label map with shape [num_classes, depth, height, width].
        """
        one_hot_map = np.zeros((num_classes,) + label_map.shape, dtype=np.float32)
        for class_idx in range(num_classes):
            one_hot_map[class_idx, ...] = (label_map == class_idx).astype(np.float32)
        return one_hot_map



    def __getitem__(self, index):
        indexF = random.randint(0,len(self.image_path)-1)
        fixed_path = self.image_path[indexF]
        fixed_labels_path = self.label_path[indexF]
        fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path))
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_labels_path))
        indexM = random.randint(0,len(self.image_path)-1)
        while(indexM==indexF):
            indexM = random.randint(0,len(self.image_path)-1)
        moving_path = self.image_path[indexM]
        moving_labels_path = self.label_path[indexM]
        moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_path))
        moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_labels_path))

        #归一化
        fixed_img = (fixed_img-fixed_img.min())/(fixed_img.max()-fixed_img.min())
        moving_img = (moving_img-moving_img.min())/(moving_img.max()-moving_img.min())

        # 扩展fixed_img和moving_img为[B,C,D,H,W]
        fixed_img = np.expand_dims(fixed_img, axis=0)
        moving_img = np.expand_dims(moving_img, axis=0)
        # 移动到GPU
        fixed_img = torch.from_numpy(fixed_img).float()
        moving_img = torch.from_numpy(moving_img).float()
        
        # fixed_label_one = self.encode_one_hot(fixed_label,12)
        # moving_label_one = self.encode_one_hot(moving_label,12)

        fixed_label = torch.from_numpy(fixed_label).int()
        fixed_label = fixed_label.type(torch.LongTensor) 
        moving_label = torch.from_numpy(moving_label).int()
        moving_label = moving_label.type(torch.LongTensor) 

        # fixed_label_one = torch.from_numpy(fixed_label_one).int()
        # fixed_label_one=  fixed_label_one.type(torch.LongTensor)
        # moving_label_one = torch.from_numpy(moving_label_one).int()
        # moving_label_one =  moving_label_one.type(torch.LongTensor)
        return {'F': fixed_img, 'M': moving_img, 'FL':fixed_label,'ML':moving_label,'name_f':self.image_path[indexF],'name_m':self.image_path[indexM]}

        # return {'F': fixed_img, 'M': moving_img, 'FL':fixed_label,'ML':moving_label,'FN':self.image_path[indexF],'MN':self.image_path[indexM],'MLO':moving_label_one,'FLO':fixed_label_one}

