为方便合并了数据集，case1-case10 为 4dct 
                   case11-case13 为 popi 
                   case14-case23 为 copd

size均为[192 160 192]
            voxelspacing 为1.6*1.6*1.6

T00被视作Fixed Image，T10-T50用作Moving Image

从每个casepack下获取mha文件的路径，命名为
case1_T10.mha
case1_T20.mha
...
case1_T50.mha
然后分为8:1:1到train,test,validation

...
从根目录下每个名称为Case{x}Pack的文件夹下的Mha文件夹下获取以.mha为后缀的文件路径，命名为Case{x}_T{y}.mha
...