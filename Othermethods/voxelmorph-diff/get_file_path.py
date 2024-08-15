import os
import glob


import random
 
 
def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    part_list = int(n_total*ratio)
    list1 = []
    list2 = []
    list1=full_list[:part_list]
    list2=full_list[part_list:]
    return list1,list2

def find_mha_files(path):
    file_path=[]
    for root,dirs,files in os.walk(path):
        if files:
            for file in files:
                if file.endswith(".nii.gz") and not file.endswith("labels.nii.gz"):
                    file_path.append(os.path.join(root,file))
    return file_path

def write_to_file(file_m,file_f,file_path):
    files = find_mha_files(file_path)
    files_m,files_f = data_split(files,0.8)
    with open(file_m, 'a+') as f:
        for file in files_m:
            f.write(f"{file}\n")
    with open(file_f, 'a+') as f:
        for file in files_f:
            f.write(f"{file}\n")
    print('Done')

path = '/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/train'
write_to_file('/home/vrdoc/GF/lung_registration/voxelmorph/MRI_M.txt','/home/vrdoc/GF/lung_registration/voxelmorph/MRI_F.txt',path)

