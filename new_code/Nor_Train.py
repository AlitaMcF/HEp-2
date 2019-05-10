##
## 该文件用于迁移网络并进行训练
##
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image as Image
from torch.autograd import Variable
import datetime
import os
import torch.optim.lr_scheduler as lr_scheduler
from ResNet50_2016 import ResNet50_2016
from ResNet50_2012 import ResNet50_2012

# *********************每次训练可能需要更改的参数都在这里*******************************
# ********************************************************************************
loopTimes = 15  # 训练的次数
modelNumFromPreTrain = 20 # 要迁移的2012预训练网络的编号
modelNumStart = 15  # 当前想基于哪个2016网络来训练
learningRate = 0.00003  # 当前网络学习率
isLoadModelFromPre = False  # 想加载的网络来自预训练的迁移还是接上上一次的训练
isUseGPU = True  # 是否使用GPU加速
path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\Hep2016"  # 数据集所在总文件夹目录
# os.path.abspath('.')当前文件目录的路径
# 模型保存所在的文件夹，不同训练模式训练出来的模型保存到不同文件夹里
preModelDir = os.path.abspath('.') + "/PreModel_2/"
norModelDir = os.path.abspath('.') + "/NorModel_1/"  # 迁移后正常训练下的网络保存的文件夹
modelNamePrefix = 'model_'  # 模型名称前缀，需要在后补上模型编号及后缀
modelNamePostfix = '.pkl'   # 模型名称后缀，即模型的格式
infoFilePrefix = 'Info_model_'
infoFilePostfix = '.log'
outputRate = 1000  # 提示显示密度，越大越高，有限制的不用担心
batchSize = 32  # 批大小，多试试几个吧
numWorker = 0  # 进程数量,先不要改这个
# ********************************************************************************
# ********************************************************************************

# 创建存放模型的文件夹
modelFolder = os.path.exists(norModelDir)
if not modelFolder:
    # 判断是否存在文件夹如果不存在则创建为文件夹
    # makedirs 创建文件时如果路径不存在会创建这个路径
    os.makedirs(norModelDir)
print('norModelFolder is at: ' + os.path.abspath(norModelDir))

# 根据机器确定是否可以使用cuda进行加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置图片加载器
def readImg(path):
    return Image.open(path).convert('L')

# 对数据集的处理规则进行设置
transform = transforms.Compose(
    [transforms.Resize((70, 70)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # 归一化公式为image=（image-mean）/std，小括号内有三个数字是对应了三通道图像
)

# 读取图片数据用于训练
trainSet = torchvision.datasets.ImageFolder(root=path + '\\train', transform=transform, loader=readImg)
print('trainSet length:' + str(len(trainSet)))
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=numWorker)
print('trainLoader length:' + str(len(trainLoader)))

if (len(trainSet) / batchSize) / outputRate < 1:
    outputRate = (len(trainSet) / batchSize)

# 根据需要加载不同网络
net = ResNet50_2016([3, 4, 6, 3], loadModelFromPre=isLoadModelFromPre, useGPU=isUseGPU)
if net.loadModelFromPre:
    modelName = preModelDir + modelNamePrefix + str(modelNumFromPreTrain) + modelNamePostfix
    # 剔除不需要的“键”
    pre_net = ResNet50_2012([3, 4, 6, 3], loadModel=True, useGPU = isUseGPU)
    pre_net.load_state_dict(torch.load(modelName))
    net_dict = net.state_dict()
    pre_dict = pre_net.state_dict()
    pre_dict = {k: v for k, v in pre_dict.items() if k in net_dict}
    net_dict.update(pre_dict)
    net.load_state_dict(net_dict)
    # # 冻结所有权重，不更新
    # for param in net.parameters():
    #     param.requires_grad = False
    # 重置最终的全连接层,重置之后最终的全连接层就可以更新权重
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, 6)
else:
    modelName = norModelDir + modelNamePrefix + str(modelNumStart) + modelNamePostfix
    net.load_state_dict(torch.load(modelName))

# 根据需要和现实情况，选择用cpu或gpu来训练
if net.useGPU is True:
    net.to(device)
    print(device)
else:
    print('cpu')

# 定义损失函数以及优化器
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, betas=(0.9, 0.999))

# 自动调节学习率
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

# 输出开始训练的时间
print('startime:' + str(datetime.datetime.now()))
startTime = datetime.datetime.now()

# 开始进行预训练网络的训练
if __name__ == '__main__':
    net.train()
    for epoch in range(loopTimes):  # 利用数据集训练loopTimes次
        # 打开或创建模型信息文件，用于保存模型运行信息
        infoFile = norModelDir + infoFilePrefix + str(modelNumStart + epoch + 1) + infoFilePostfix
        running_loss = 0.0
        running_corrects = 0

        with open(infoFile, 'a+') as f:
            f.write(
                'batchsize: ' + str(batchSize) + '\n' + 'optim: ' + str(
                    optimizer) + '\n')

        for batchIndex, data in enumerate(trainLoader, 0):
            # 得到输入数据
            inputs, labels = data
            # 包装数据
            inputs, labels = Variable(inputs), Variable(labels)
            # 根据需要将数据加载到GPU中
            if net.useGPU is True:
                inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播，优化
            outputs = net(inputs)

            # 批测试大概的正确率
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 损失值计算
            running_loss += loss.data * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 依据密度来输出实时数据
            if (batchIndex + 1) % int(len(trainLoader) / outputRate) == 0:
                print('epoch: %d [%.1f%%]  loss:%.5f  acc:%.5f%%' %
                      (epoch + 1, (batchIndex + 1) / len(trainLoader) * 100,
                       running_loss / ((batchIndex + 1) * batchSize),
                       running_corrects.item() * 100 / (
                               (batchIndex + 1) * batchSize)))

                with open(infoFile, 'a+') as f:
                    f.write('epoch: %d [%.1f%%]  loss:%.5f  acc:%.5f%%' %
                            (epoch + 1,
                             (batchIndex + 1) / len(trainLoader) * 100,
                             running_loss / ((batchIndex + 1) * batchSize),
                             running_corrects.item() * 100 / (
                                     (batchIndex + 1) * batchSize)) + '\n')

            # 计算并输出预计完成时间,每10行输出一次
            if (batchIndex + 1) % int(len(trainLoader) / outputRate * 10) == 0:
                currentTime = datetime.datetime.now()
                print('estimated finish time: ' +
                        str(startTime + (
                                currentTime - startTime) / (batchIndex + 1) * len(
                            trainLoader)))
                print('estimated finish all time: ' +
                        str(startTime + (
                                currentTime - startTime) / (batchIndex + 1) * len(
                            trainLoader) * (loopTimes - epoch)))

        # # 保存网络
        # savedModleName = norModelDir + modelNamePrefix + str(modelNumStart + epoch) + modelNamePostfix
        # torch.save(net.state_dict(), savedModleName)

        # 保存网络
        savedModelName = norModelDir + modelNamePrefix + str(
            modelNumStart + epoch + 1) + modelNamePostfix
        torch.save(net.state_dict(), savedModelName)

        # 输出每一轮训练结束的时间，便于统计训练时长
        print('finish ' + str(epoch+1) + ': ' + str(datetime.datetime.now()))
        with open(infoFile, 'a+') as f:
            f.write('finish ' + str(epoch+1) + ': ' + str(datetime.datetime.now()) + '\n')
            f.write('runtime ' + str(datetime.datetime.now() - startTime) + '\n')

        # 学习率step
        exp_lr_scheduler.step()
