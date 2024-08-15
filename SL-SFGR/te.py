import numpy as np

# 假设文件名为 'data.npy'
filename = '/root/autodl-tmp/OASIS3_Dataset/train/image_pairs.npy'

# 读取 .NPY 文件
data = np.load(filename)

# 打印读取到的数据
print(len(data))
