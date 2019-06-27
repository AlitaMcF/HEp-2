import random
import os
import shutil

ratio = 0.2  # 测试集占原数据的比例
path = "D:\DataAndHomework\HEp-2细胞项目\数据集\Hep2016"  # 总文件夹目录

trainPath = path + '/train/Centromere/'
testPath = path + '/test/Centromere/'
files = os.listdir(trainPath)

length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
length = int(length * ratio)
test = random.sample(files, length)
for name in test:
    shutil.move(trainPath + name, testPath + name)
print(trainPath + '    is DONE!')
print(str(length) + '   files!')

trainPath = path + '/train/Golgi/'
testPath = path + '/test/Golgi/'
files = os.listdir(trainPath)

length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
length = int(length * ratio)
test = random.sample(files, length)
for name in test:
    shutil.move(trainPath + name, testPath + name)
print(trainPath + '    is DONE!')
print(str(length) + '   files!')

trainPath = path + '/train/Homogeneous/'
testPath = path + '/test/Homogeneous/'
files = os.listdir(trainPath)

length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
length = int(length * ratio)
test = random.sample(files, length)
for name in test:
    shutil.move(trainPath + name, testPath + name)
print(trainPath + '    is DONE!')
print(str(length) + '   files!')

trainPath = path + '/train/Nucleolar/'
testPath = path + '/test/Nucleolar/'
files = os.listdir(trainPath)

length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
length = int(length * ratio)
test = random.sample(files, length)
for name in test:
    shutil.move(trainPath + name, testPath + name)
print(trainPath + '    is DONE!')
print(str(length) + '   files!')

trainPath = path + '/train/NuMem/'
testPath = path + '/test/NuMem/'
files = os.listdir(trainPath)

length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
length = int(length * ratio)
test = random.sample(files, length)
for name in test:
    shutil.move(trainPath + name, testPath + name)
print(trainPath + '    is DONE!')
print(str(length) + '   files!')

trainPath = path + '/train/Speckled/'
testPath = path + '/test/Speckled/'
files = os.listdir(trainPath)

length = len([name for name in os.listdir(trainPath) if os.path.isfile(os.path.join(trainPath, name))])
length = int(length * ratio)
test = random.sample(files, length)
for name in test:
    shutil.move(trainPath + name, testPath + name)
print(trainPath + '    is DONE!')
print(str(length) + '   files!')
