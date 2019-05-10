import torch
import torchvision
import numpy
from PIL import Image
import os

# *****************************************************************************
# 负责进行预处理，依据文件夹里的原图和mask进行处理得到新的图存入文件夹里
# 也就是将遮罩打上
# *****************************************************************************

path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\Hep2016"  # 数据集所在总文件夹目录
cellType = ['Centromere', 'Golgi', 'Homogeneous', 'Nucleolar', 'NuMem', 'Speckled']  # 细胞种类名称

for ctype in cellType:

    sourcePath = path + '/source/' + str(ctype) + '/'
    dstPath = path + '/afterMask/' + str(ctype) + '/'
    oriFolder = os.path.exists(sourcePath)
    dstFolder = os.path.exists(dstPath)

    if not oriFolder:  # 判断是否存在文件夹如果不存在则结束程序
        print('sourceFolder:  '+sourcePath + " not found!")
        exit(-1)
    else:
        print('sourceFolder:  '+sourcePath + " found!")

    if not dstFolder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(dstPath)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print('dstFolder:  '+dstPath + " created!")
    else:
        print('dstFolder:  '+dstPath + " found!")

    oriFiles = os.listdir(sourcePath)
    iterFile = iter(oriFiles)
    fileNum = len(oriFiles);
    if fileNum % 2 != 0:
        print("sourceFile's total number is error!: "+fileNum)
        exit(-1)
    fileNum=int(fileNum/2)

    for index in range(0, fileNum):  # 遍历文件夹
        # 读取原图
        file = next(iterFile)
        fileName=str(file)
        im1 = Image.open(sourcePath + file)
        im1N = numpy.array(im1)
        # 读取mask
        file = next(iterFile)
        im2 = Image.open(sourcePath + file)
        im2N = numpy.array(im2)
        im2N = im2N & 1  # 化为1的掩码
        # 制作处理后的图
        finalN = im1N * im2N  # 乘掩码删去无关部分
        # 转换回image,数据本身就是灰度图L
        final = Image.fromarray(finalN.astype('uint8')).convert('L')
        # 保存为新图放入新准备的文件夹中
        final.save(dstPath + fileName)
        # 进度
        if index % 500 == 0:
            print(fileName)
    print('\n')
