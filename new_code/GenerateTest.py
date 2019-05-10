import random
import os
import shutil

# *****************************************************************************
# 将Mask后的图片读取并随机分割为测试集和训练集
# 是否需要交叉验证集？
# *****************************************************************************

ratio = 0.2  # 用作测试集的比例
path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\Hep2016"  # 数据集所在总文件夹目录
cellType = ['Centromere', 'Golgi', 'Homogeneous', 'Nucleolar', 'NuMem', 'Speckled']  # 细胞种类名称

for ctype in cellType:

    sourcePath = path + '/afterMask/' + str(ctype) + '/'
    trainPath = path + '/train/' + str(ctype) + '/'
    testPath = path + '/test/' + str(ctype) + '/'
    oriFolder = os.path.exists(sourcePath)
    trainFolder = os.path.exists(trainPath)
    testFolder = os.path.exists(testPath)

    if not oriFolder:  # 判断是否存在文件夹如果不存在则结束程序
        print('sourceFolder:  ' + sourcePath + " not found!")
        exit(-1)
    else:
        print('sourceFolder:  ' + sourcePath + " found!")

    if not trainFolder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(trainPath)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print('trainFolder:  ' + trainPath + " created!")
    else:
        print('trainFolder:  ' + trainPath + " found!")

    if not testFolder:
        os.makedirs(testPath)
        print('testFolder:  ' + testPath + " created!")
    else:
        print('testFolder:  ' + testPath + " found!")

    oriFiles = os.listdir(sourcePath)
    for file in oriFiles:
        shutil.copyfile(sourcePath + file, trainPath + file)  # 复制文件
    print("files copy is DONE!")

    trainFiles = os.listdir(trainPath)
    length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
    # 获取需要的抽取数量
    length = int(length * ratio)
    # 随机抽取
    test = random.sample(trainFiles, length)
    for name in test:
        shutil.move(trainPath + name, testPath + name)
    print("files segmentation is DONE!")
    print(str(length) + ' files in testFolder\n')
