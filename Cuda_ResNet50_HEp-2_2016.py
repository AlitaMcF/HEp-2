import numpy as np
import torch.nn as nn
import torch
import math
import torchvision.transforms as transforms
import torchvision
from PIL import Image as Image
import matplotlib.pyplot as plt
from torch.autograd import Variable

path = "D:\\DataAndHomework\\HEp-2细胞项目\\数据集\\Hep2016"  # 总文件夹目录

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    expansion = 4

    def __init__(self, inChannel, outChannel, stride = 1, decSample = None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(outChannel)
        self.conv3 = nn.Conv2d(outChannel, outChannel*self.expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(outChannel*4)
        self.relu = nn.ReLU(inplace = True)
        self.decSample = decSample # 下采样层
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.decSample is not None:
            residual = self.decSample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet50(nn.Module):
    def __init__(self, layers, num_classes = 6, model_path = 'HEp-2_ResNet50_Model.pkl', loadModel = False):
        super(ResNet50, self).__init__()
        self.inChannel = 64
        self.modelPath = model_path
        self.loadModel = loadModel
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.block1 = self.make_block(64, layers[0])
        self.block2 = self.make_block(128, layers[1], stride = 2)
        self.block3 = self.make_block(256, layers[2], stride = 2)
        self.block4 = self.make_block(512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(3, stride = 1) # 3×3 池化层
        self.linear = nn.Linear(512*Residual_Block.expansion, num_classes)
        self.init_param()


    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()


    # 创建残差网络的"块"
    def make_block(self, channel, blocks, stride = 1):
        decSample = None
        layers = []

        # 创建下采样层
        if stride != 1 or self.inChannel != channel*Residual_Block.expansion:
            decSample = nn.Sequential(
                nn.Conv2d(self.inChannel, channel * Residual_Block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(channel * Residual_Block.expansion)
            )

        layers.append(Residual_Block(self.inChannel, channel, stride, decSample))
        self.inChannel = channel*Residual_Block.expansion
        for i in range(1, blocks):
            layers.append(Residual_Block(self.inChannel, channel))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x



# 对数据集进行处理
transform = transforms.Compose(
    [transforms.Resize((70, 70)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 读取图片数据
trainSet = torchvision.datasets.ImageFolder(root=path + '/train', transform=transform, loader=readImg)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1,shuffle=True, num_workers=0)
testSet = torchvision.datasets.ImageFolder(root=path + '/test', transform=transform, loader=readImg)
testloader = torch.utils.data.DataLoader(testSet, batch_size=1,shuffle=False, num_workers=0)
classes = testSet.classes

# 创建网络
net = ResNet50([3, 4, 6, 3], loadModel = False)
if net.loadModel:
    net.load_state_dict(torch.load('HEp-2_ResNet50_Model.pkl'))

# 将网络加载到GPU中
net.to(device)

# 定义损失函数以及优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# 训练网络
for epoch in range(2):  # 利用数据集训练两次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 得到输入数据
        inputs, labels = data
        # 包装数据
        inputs, labels = Variable(inputs), Variable(labels)
        # 再将数据加载到GPU中
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 输出损失值
        running_loss += loss.data
        if i % 2 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2))
            running_loss = 0.0

print('Finished Training')

# 统计训练效果
dataiter = iter(testloader)
Sum = 0
TruePrd = 0
while True:
    try:
        images, labels = dataiter.next()
        Sum += 1
        print('GroundTruth:', ' '.join('%5s'%(classes[labels[0]])))

        # 输出神经网络的分类效果
        outputs = net(Variable(images))

        # 获取6个类别的预测值大小，预测值越大，神经网络认为属于该类别的可能性越大
        _, predicted = torch.max(outputs.data, 1)

        print('Predicted:', ' '.join('%5s'%(classes[predicted[0]])))

        if labels[0] == predicted[0]:
            TruePrd = TruePrd + 1
    except StopIteration:
        break

print(TruePrd)
print(Sum)
print(float(TruePrd)/Sum)

# 保存网络
torch.save(net.state_dict(), 'HEp-2_ResNet50_Model.pkl')