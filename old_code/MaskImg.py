import torch
import torchvision
import numpy
from PIL import Image
import os

# # 预处理,依据文件夹里的原图和mask进行处理得到新的图存入文件夹里
path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\Hep2016"  # 总文件夹目录

files = os.listdir(path + '/source/Centromere/')  # 得到目标文件夹(train)下的所有文件名称
iterFile = iter(files)
for index in range(7924, 10664):  # 遍历文件夹
    # 读取原图
    file = next(iterFile)
    im1 = Image.open(path + '/source/Centromere/' + file)
    im1N = numpy.array(im1)
    # 读取mask
    file = next(iterFile)
    im2 = Image.open(path + '/source/Centromere/' + file)
    im2N = numpy.array(im2)
    im2N = im2N & 1  # 化为1的掩码
    # 制作处理后的图
    finalN = im1N * im2N  # 乘掩码删去无关部分
    final = Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    final.save(path + '/afterMask/Centromere/' + str(index) + '.png')  # 保存为新图放入新准备的文件夹中
    # 进度
    if index % 500 == 0:
        print(index)

files = os.listdir(path + '/source/Golgi/')  # 得到目标文件夹(train)下的所有文件名称
iterFile = iter(files)
for index in range(12873, 13596):  # 遍历文件夹
    # 读取原图
    file = next(iterFile)
    im1 = Image.open(path + '/source/Golgi/' + file)
    im1N = numpy.array(im1)
    # 读取mask
    file = next(iterFile)
    im2 = Image.open(path + '/source/Golgi/' + file)
    im2N = numpy.array(im2)
    im2N = im2N & 1  # 化为1的掩码
    # 制作处理后的图
    finalN = im1N * im2N  # 乘掩码删去无关部分
    final = Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    final.save(path + '/afterMask/Golgi/' + str(index) + '.png')  # 保存为新图放入新准备的文件夹中
    # 进度
    if index % 500 == 0:
        print(index)

files = os.listdir(path + '/source/Homogeneous/')  # 得到目标文件夹(train)下的所有文件名称
iterFile = iter(files)
for index in range(1, 2494):  # 遍历文件夹
    # 读取原图
    file = next(iterFile)
    im1 = Image.open(path + '/source/Homogeneous/' + file)
    im1N = numpy.array(im1)
    # 读取mask
    file = next(iterFile)
    im2 = Image.open(path + '/source/Homogeneous/' + file)
    im2N = numpy.array(im2)
    im2N = im2N & 1  # 化为1的掩码
    # 制作处理后的图
    finalN = im1N * im2N  # 乘掩码删去无关部分
    final = Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    final.save(path + '/afterMask/Homogeneous/' + str(index) + '.png')  # 保存为新图放入新准备的文件夹中
    # 进度
    if index % 500 == 0:
        print(index)

files = os.listdir(path + '/source/Nucleolar/')  # 得到目标文件夹(train)下的所有文件名称
iterFile = iter(files)
for index in range(5326, 7923):  # 遍历文件夹
    # 读取原图
    file = next(iterFile)
    im1 = Image.open(path + '/source/Nucleolar/' + file)
    im1N = numpy.array(im1)
    # 读取mask
    file = next(iterFile)
    im2 = Image.open(path + '/source/Nucleolar/' + file)
    im2N = numpy.array(im2)
    im2N = im2N & 1  # 化为1的掩码
    # 制作处理后的图
    finalN = im1N * im2N  # 乘掩码删去无关部分
    final = Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    final.save(path + '/afterMask/Nucleolar/' + str(index) + '.png')  # 保存为新图放入新准备的文件夹中
    # 进度
    if index % 500 == 0:
        print(index)

files = os.listdir(path + '/source/NuMem/')  # 得到目标文件夹(train)下的所有文件名称
iterFile = iter(files)
for index in range(10665, 12872):  # 遍历文件夹
    # 读取原图
    file = next(iterFile)
    im1 = Image.open(path + '/source/NuMem/' + file)
    im1N = numpy.array(im1)
    # 读取mask
    file = next(iterFile)
    im2 = Image.open(path + '/source/NuMem/' + file)
    im2N = numpy.array(im2)
    im2N = im2N & 1  # 化为1的掩码
    # 制作处理后的图
    finalN = im1N * im2N  # 乘掩码删去无关部分
    final = Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    final.save(path + '/afterMask/NuMem/' + str(index) + '.png')  # 保存为新图放入新准备的文件夹中
    # 进度
    if index % 500 == 0:
        print(index)

files = os.listdir(path + '/source/Speckled/')  # 得到目标文件夹(train)下的所有文件名称
iterFile = iter(files)
for index in range(2495, 5325):  # 遍历文件夹
    # 读取原图
    file = next(iterFile)
    im1 = Image.open(path + '/source/Speckled/' + file)
    im1N = numpy.array(im1)
    # 读取mask
    file = next(iterFile)
    im2 = Image.open(path + '/source/Speckled/' + file)
    im2N = numpy.array(im2)
    im2N = im2N & 1  # 化为1的掩码
    # 制作处理后的图
    finalN = im1N * im2N  # 乘掩码删去无关部分
    final = Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    final.save(path + '/afterMask/Speckled/' + str(index) + '.png')  # 保存为新图放入新准备的文件夹中
    # 进度
    if index % 500 == 0:
        print(index)
