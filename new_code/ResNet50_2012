##
## 该文件定义是用于2012数据集的预训练网络结构
##
import torch.nn as nn
import math

# 残差块类
class Residual_Block(nn.Module):
    expansion = 4

    def __init__(self, inChannel, outChannel, stride=1, decSample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=1, bias=True) # 卷积层
        self.bn1 = nn.BatchNorm2d(outChannel)   # 归一化层
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(outChannel)
        self.conv3 = nn.Conv2d(outChannel, outChannel * self.expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(outChannel * 4)
        self.relu = nn.ReLU(inplace=True)   # 激活层
        self.decSample = decSample  # 下采样层
        self.stride = stride    # 步长
        self.dropout = nn.Dropout(0.15) # 随机将张量中的部分元素置0，概率为参数对应的值

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)    # 每个卷积和归一化层后带一个激活层

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


# ResNet50网络类
class ResNet50_2012(nn.Module):
    def __init__(self, layers, num_classes=6, model_path='model.pkl', loadModel=False, useGPU=True):
        super(ResNet50_2012, self).__init__()
        self.inChannel = 64
        self.modelPath = model_path
        self.useGPU = useGPU
        self.loadModel = loadModel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True) # 与ResNet50_2016不同的地方在于一开始输入的通道数，这里是彩色图像，所以是3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self.make_block(64, layers[0])
        self.block2 = self.make_block(128, layers[1], stride=2)
        self.block3 = self.make_block(256, layers[2], stride=2)
        self.block4 = self.make_block(512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)  # 3×3 池化层
        self.linear = nn.Linear(512 * Residual_Block.expansion, num_classes)
        self.init_param()

    # 初始化参数
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    # 创建残差网络的"块"
    def make_block(self, channel, blocks, stride=1):
        decSample = None
        layers = []

        # 创建下采样层
        if stride != 1 or self.inChannel != channel * Residual_Block.expansion:
            decSample = nn.Sequential(
                nn.Conv2d(self.inChannel, channel * Residual_Block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * Residual_Block.expansion)
            )

        layers.append(Residual_Block(self.inChannel, channel, stride, decSample))
        self.inChannel = channel * Residual_Block.expansion
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
