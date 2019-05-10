import os
import numpy as np
from PIL import Image

# *****************************************************************************
# 负责扩展已经分割好的训练集，进行翻转操作和添加噪声
# *****************************************************************************

path = "E:/学校学习/导师/HEp-2细胞数据集/dataset/Hep2016/"  # 总训练集的文件目录
cellType = ['Centromere', 'Golgi', 'Homogeneous', 'Nucleolar', 'NuMem', 'Speckled']  # 细胞种类名称
X = -1  # 椒盐数量,-1不进行

for ctype in cellType:

    trainPath = path + '/train/' + str(ctype) + '/'
    ex_trainPath = path + '/ex_train/' + str(ctype) + '/'
    oriFolder = os.path.exists(trainPath)
    dstFolder = os.path.exists(ex_trainPath)

    if not dstFolder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs(ex_trainPath)
        print(ex_trainPath + " created!")
    else:
        print(ex_trainPath + " found!")

    oriFiles = os.listdir(trainPath)
    count = 0
    for file in oriFiles:  # 遍历文件夹
        count += 1
        if count % 200 == 0:
            print(file)
        file_path = os.path.join(trainPath, file)
        img = Image.open(file_path)
        img.save(ex_trainPath + 'Ori_' + file)  # 拷贝原始图像

        dst1 = img.transpose(Image.FLIP_LEFT_RIGHT)  # 左右互换
        dst1.save(ex_trainPath + 'LR_' + file)

        dst2 = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下互换
        dst2.save(ex_trainPath + 'TD_' + file)

        dst3 = dst1.transpose(Image.FLIP_TOP_BOTTOM)  # 左右上下互换
        dst3.save(ex_trainPath + 'All_' + file)

        if X != -1:
            dst4 = np.array(img)  # 椒盐图像
            rows, cols = dst4.shape
            for i in range(X):
                x = np.random.randint(0, rows)
                y = np.random.randint(0, cols)
                dst4[x, y] = 255
            dst4.flags.writeable = True
            dst4 = Image.fromarray(np.uint8(dst4))
            dst4.save(ex_trainPath + 'Nos_' + file)
