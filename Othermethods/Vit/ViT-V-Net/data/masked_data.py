import os
import glob
import SimpleITK as sitk
def find_mha_files(directory):
    # 构建匹配模式
    pattern = os.path.join(directory, '*.mha')
    # 使用 glob 模块匹配文件
    mha_files = glob.glob(pattern)
    return mha_files
def construct_mask_filepath(original_filepath, maskdata_dir):
    # 获取原始文件名（不含路径）
    original_filename = os.path.basename(original_filepath)
    # 分割文件名和扩展名
    filename_parts = os.path.splitext(original_filename)
    # 分割文件名的 case 和 time_point 部分
    case, time_point = filename_parts[0].split('_')
    if time_point =='T0':
        time_point='T00'
    # 构建对应的 mask 文件名
    case = 'C'+case[case.find('case')+5:]
    mask_filename = f'{case}{time_point}_lungmask.mha'

    # 构建对应的 mask 文件路径
    mask_filepath = os.path.join(maskdata_dir, mask_filename)
    mask_filename_new = f'{case}{time_point}_lungmask.nii.gz'
    return mask_filepath,mask_filename_new 
path_M = '/home/vrdoc/GF/lung_registration/data/train/Moving/'
path_F = '/home/vrdoc/GF/lung_registration/data/train/Fixed/'
mask_dir = '/home/vrdoc/GF/lung_registration/data/maskdata/'
movings = find_mha_files(path_M)
fixeds = find_mha_files(path_F)
for moving in movings:
    mask,name = construct_mask_filepath(moving,mask_dir)
    mo0 = sitk.ReadImage(moving)
    mo = sitk.GetArrayFromImage(mo0)
    ma0 = sitk.ReadImage(mask)
    ma = sitk.GetArrayFromImage(ma0)
    res = mo*ma
    res = (res - res.min())/res.max()
    res = sitk.GetImageFromArray(res)
    res.SetDirection(mo0.GetDirection())
    res.SetOrigin(mo0.GetOrigin())
    res.SetSpacing(mo0.GetSpacing())
    sitk.WriteImage(res,'/home/vrdoc/GF/lung_registration/data/maskeddata/'+name)

for moving in fixeds:
    mask,name = construct_mask_filepath(moving,mask_dir)
    mo0 = sitk.ReadImage(moving)
    mo = sitk.GetArrayFromImage(mo0)
    ma0 = sitk.ReadImage(mask)
    ma = sitk.GetArrayFromImage(ma0)
    res = mo*ma
    res = (res - res.min())/res.max()
    res = sitk.GetImageFromArray(res)
    res.SetDirection(mo0.GetDirection())
    res.SetOrigin(mo0.GetOrigin())
    res.SetSpacing(mo0.GetSpacing())
    sitk.WriteImage(res,'/home/vrdoc/GF/lung_registration/data/maskeddata/'+name)
