# -*- coding: utf-8 -*- 
# @Time : 2019-11-19 16:15 
# @Author : Trible 

from torch import nn
import torch.nn.functional as F
import torch
import math
from collections import OrderedDict

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dim):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 32], layers[0], 0)
        self.layer2 = self._make_layer([32, 64], layers[1], 1)
        self.layer3 = self._make_layer([64, 64], layers[2], 2)
        self.layer4 = self._make_layer([64, 128], layers[3], 3)
        self.layer5 = self._make_layer([128, 128], layers[4], 4)

        self.linear_layer = nn.Linear(128 * 4 * 4, 512)
        self.output_layer = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, layer_num):
        layers = []

        dim = 208
        if layer_num == 0:
            dim = 416
        elif layer_num == 1:
            dim = 208
        elif layer_num == 2:
            dim = 104
        elif layer_num == 3:
            dim = 52
        elif layer_num == 4:
            dim = 26
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        #  blocks
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes, dim // 2)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        #
        fc = out5.reshape(-1, 128 * 4 * 4)
        feature = self.linear_layer(fc)
        output = self.output_layer(feature)
        return feature, output

if __name__ == "__main__":
    net = DarkNet([1, 1, 2, 2, 1]).cuda()
    x = torch.randn((1, 3, 128, 128)).cuda()
    f, ys= net(x)
    print(f.size())
    print(ys.size())