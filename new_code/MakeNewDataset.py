import os
import shutil

import scipy.io as scio

path = "E:/学校学习/导师/HEp-2细胞数据集/dataset/HepLarge/"  # 总文件夹目录
labelType = ['1', '2', '3', '4', '5', '6']  # label名

sourcePath = path + '/cells/cells/'

oriFolder = os.path.exists(sourcePath)

if not oriFolder:  # 判断是否存在文件夹如果不存在则结束程序
    print('sourceFolder:  ' + sourcePath + " not found!")
    exit(-1)
else:
    print('sourceFolder:  ' + sourcePath + " found!")

data = scio.loadmat(path + '/labels.mat')
a = data['labels']
a = a[0]
print(len(a))

for index in range(0, len(a)):
    dstPath = path + '/source/' + str(a[index]) + '/'
    dstFolder = os.path.exists(dstPath)
    if not dstFolder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(dstPath)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print('dstFolder:  ' + dstPath + " created!")
    shutil.copyfile(sourcePath + str(index + 1) + '.png', dstPath + str(index + 1) + '.png')  # 复制文件
    if index % 500 == 0:
        print(index)
