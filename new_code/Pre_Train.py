#
# 该文件用于预训练网络的训练
#
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image as Image
from torch.autograd import Variable
from datetime import datetime, timedelta
import os
from ResNet50_2012 import ResNet50_2012
import torch.optim.lr_scheduler as lr_scheduler

# *********************每次训练可能需要更改的参数都在这里*******************************
# ********************************************************************************
loopTimes = 5  # 训练的次数
modelNumStart = 15  # 当前想基于哪个网络来训练,开始时填0
learningRate = 0.0001  # 当前网络学习率,默认Adam是千分之一
isLoadModel = True  # 是否加载已有的网络
isUseGPU = True  # 是否使用GPU加速
path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\HepLarge"  # 数据集所在总文件夹目录
# os.path.abspath('.')当前文件目录的路径
# 模型保存所在的文件夹，不同训练模式训练出来的模型保存到不同文件夹里
modelDir = os.path.abspath('.') + "/PreModel_2/"
modelNamePrefix = 'model_'  # 模型名称前缀，需要在后补上模型编号及后缀
modelNamePostfix = '.pkl'  # 模型名称后缀，即模型的格式
infoFilePrefix = 'info_model_'
infoFilePostfix = '.log'
outputRate = 1000  # 提示显示密度，越大越高，有限制的不用担心
batchSize = 64  # 批大小
numWorker = 0  # 进程数量,先不要改这个
# ********************************************************************************
# ********************************************************************************

# 创建存放模型的文件夹
modelFolder = os.path.exists(modelDir)
if not modelFolder:
    # 判断是否存在文件夹如果不存在则创建为文件夹 
    # makedirs 创建文件时如果路径不存在会创建这个路径
    os.makedirs(modelDir)
print('modelFolder is at: ' + os.path.abspath(modelDir))

# 根据机器确定是否可以使用cuda进行加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 设置图片加载器
def readImg(path):
    return Image.open(path)


# 对数据集的处理规则进行设置
transform = transforms.Compose(
    [transforms.Resize((70, 70)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # 归一化公式为image=（image-mean）/std，小括号内有三个数字是对应了三通道图像
)

# 读取图片数据用于训练
trainSet = torchvision.datasets.ImageFolder(root=path + '/source',
                                            transform=transform, loader=readImg)
print('trainSet length:' + str(len(trainSet)))
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize,
                                          shuffle=True,
                                          num_workers=numWorker)
print('trainLoader length:' + str(len(trainLoader)))

if (len(trainSet) / batchSize) / outputRate < 1:
    outputRate = (len(trainSet) / batchSize)

# 创建网络
net = ResNet50_2012([3, 4, 6, 3], loadModel=isLoadModel, useGPU=isUseGPU)
if net.loadModel:
    modelName = modelDir + modelNamePrefix + str(
        modelNumStart) + modelNamePostfix
    net.load_state_dict(torch.load(modelName))

# 根据需要和现实情况，选择用cpu或gpu来训练
if net.useGPU is True:
    net.to(device)
    print(device)
else:
    print('cpu')

# 定义损失函数以及优化器
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate,
                             betas=(0.9, 0.999))

# 自动调节学习率
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

# 输出开始训练的时间
print('startime:' + str(datetime.now()))
startTime = datetime.now()

# 开始进行预训练网络的训练
if __name__ == '__main__':
    net.train(True)

    for epoch in range(loopTimes):  # 利用数据集训练loopTimes次
        # 打开或创建模型信息文件，用于保存模型运行信息
        infoFile = modelDir + infoFilePrefix + str(
            modelNumStart + epoch + 1) + infoFilePostfix
        running_loss = 0.0
        running_corrects = 0

        with open(infoFile, 'a+') as f:
            f.write(
                'batchsize: ' + str(batchSize) + '\n' + 'optim: ' + str(
                    optimizer) + '\n')

        for batchIndex, data in enumerate(trainLoader):
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

            _, preds = torch.max(outputs.data, 1)  # 返回每一行的最大值的内容,即其预测

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 损失值统计
            running_loss += loss.data * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)  # 预测==标签的个数

            # 依据密度来输出实时数据
            if (batchIndex + 1) % int(len(trainLoader) / outputRate) == 0:
                print('epoch: %d [%.1f%%]  loss:%.5f  acc:%.5f%%' %
                      (epoch + 1, (batchIndex + 1) / len(trainLoader) * 100,
                       running_loss / ((batchIndex + 1) * batchSize),
                       running_corrects.item() * 100 / (
                               (batchIndex + 1) * batchSize)))

                # 可能是由于我们提前结束输出，文件之前没能到达close处，用with我记得会自动处理close
                with open(infoFile, 'a+') as f:
                    f.write('epoch: %d [%.1f%%]  loss:%.5f  acc:%.5f%%' %
                            (epoch + 1,
                             (batchIndex + 1) / len(trainLoader) * 100,
                             running_loss / ((batchIndex + 1) * batchSize),
                             running_corrects.item() * 100 / (
                                     (batchIndex + 1) * batchSize)) + '\n')

            # 计算并输出预计完成时间,每10行输出一次
            if (batchIndex + 1) % int(len(trainLoader) / outputRate * 10) == 0:
                currentTime = datetime.now()
                print('estimated finish time: ' +
                      str(startTime + (
                              currentTime - startTime) / (batchIndex + 1) * len(
                          trainLoader)))
                print('estimated finish all time: ' +
                      str(startTime + (
                              currentTime - startTime) / (batchIndex + 1) * len(
                          trainLoader) * (loopTimes - epoch)))

        # 保存网络
        savedModelName = modelDir + modelNamePrefix + str(
            modelNumStart + epoch + 1) + modelNamePostfix
        torch.save(net.state_dict(), savedModelName)

        # 输出每一轮训练结束的时间，便于统计训练时长
        print('finish ' + str(epoch + 1) + ': ' + str(datetime.now()))
        with open(infoFile, 'a+') as f:
            f.write(
                'finish ' + str(epoch + 1) + ': ' + str(datetime.now()) + '\n')
            f.write('runtime ' + str(datetime.now() - startTime) + '\n')

        # 学习率step
        exp_lr_scheduler.step()

# 感觉多输出写数据比较好
