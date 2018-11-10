import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image as Image
import numpy as np
import torchvision


path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\Hep2016"  # 总文件夹目录

def readImg(path):
    return Image.open(path)

# 展示图片
def imshow(img):
    img = img / 2 + 0.5  # 非标准化
    npimg = img.numpy()
    # Image.fromarray(finalN.astype('uint8')).convert('L')  # 转换回image,数据本身就是灰度图L
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, same_shape = True):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel/4, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel/4, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channel*4)
        self.relu = nn.ReLU(inplace = True)
        self.same_shape = same_shape
        if same_shape == False:
            self.decSample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel*4)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if not self.same_shape:
            x = self.decSample(x)

        out += x
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, layers, num_classes = 6, model_path = 'resnet50.pkl'):
        super(ResNet50, self).__init__()
        self.model_path = model_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.cur_channel = 64
        self.stack1 = self.make_stack(256, layers[0], same_channel= True)
        self.stack2 = self.make_stack(512, layers[1], stride = 2)
        self.stack3 = self.make_stack(1024, layers[2], stride = 2)
        self.stack4 = self.make_stack(2048, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.classifier = nn.linear(2048, num_classes) # 这里有点问题

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def make_stack(self, out_channel, blocks, stride = 1, same_channel = False):
        layers = []
        layers.append(Residual_Block(self.cur_channel, out_channel, stride, same_channel))
        self.cur_channel = out_channel
        for i in range(2, blocks):
            layers.append(Residual_Block(self.cur_channel, out_channel))

        return nn.Sequential(*layers)



# 对数据集进行处理
transform = transforms.Compose(
    [transforms.Resize((70, 70)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainSet = torchvision.datasets.ImageFolder(root=path + '/afterMask', transform=transform, loader=readImg)

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1,
                                          shuffle=True, num_workers=0)

testSet = torchvision.datasets.ImageFolder(root=path + '/test', transform=transform, loader=readImg)

testloader = torch.utils.data.DataLoader(testSet, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = testSet.classes




net = ResNet50([3, 4, 6, 3]);

# 定义损失函数以及优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# 训练网络
for epoch in range(2):  # 利用数据集训练两次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 得到输入数据
        inputs, labels = data

        # 包装数据
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # display information
        running_loss += loss.data
        if i % 2000 == 0:
            print('[%d, %5d] loss: %.3f' .format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
Sum = 0
TruePrd = 0
while True:
    try:
        images, labels = dataiter.next()
        Sum += 1
        print('GroundTruth:', ' '.join('%5s'.format(classes[labels[0]])))

        # 输出神经网络的分类效果
        outputs = net(Variable(images))

        # 获取6个类别的预测值大小，预测值越大，神经网络认为属于该类别的可能性越大
        _, predicted = torch.max(outputs.data, 1)

        print('Predicted:', ' '.join('%5s'.format(classes[predicted[0]])))

        if labels[0] == predicted[0]:
            TruePrd = TruePrd + 1
    except StopIteration:
        break

print(TruePrd)
print(Sum)
print(float(TruePrd)/Sum)
