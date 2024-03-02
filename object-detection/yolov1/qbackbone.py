import brevitas.nn as qnn

from torch import nn
from torch.nn import Module
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, weight_bit_width=4)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=4)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, weight_bit_width=4)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, weight_bit_width=4)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = qnn.QuantConv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False, weight_bit_width=4)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):

        super().__init__()
        self.in_channels = 64
        self.conv1  = qnn.QuantConv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_bit_width=4)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #initializing the weights of the Conv2d
        #and batchnorm layers
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if m in self.modules():
            if isinstance(m, BottleNeck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
        '''

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(qnn.QuantConv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False, weight_bit_width=4), nn.BatchNorm2d(out_channels*block.expansion),)

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        C1 = self.conv1(x)
        C1 = self.bn1(C1)
        C1 = self.relu(C1)
        C1 = self.maxpool(C1)

        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        return C5


if __name__ == "__main__":
    import torch
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    print(model)
    
    input_size = 128
    x = torch.randn(1, 1, input_size, input_size).to('cpu')
    y = model(x)
    print(y.shape)
