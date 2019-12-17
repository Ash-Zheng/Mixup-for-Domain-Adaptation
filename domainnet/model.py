import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate= 0.0, activate_before_residual= False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum= 0.001)
        self.relu1 = nn.LeakyReLU(negative_slope= 0.1, inplace= True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size= 3, stride= stride, padding= 1, bias= False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum= 0.001)
        self.relu2 = nn.LeakyReLU(negative_slope= 0.1, inplace= True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.dropRate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortCut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size= 1, stride= stride,
                                                                padding= 0, bias= False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropRate > 0:
            out = F.dropout(out, p= self.dropRate, training= self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortCut(x), out)

class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate= 0.0, activate_before_residual= False):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, input):
        return self.layer(input)

class WideResNetFeature(nn.Module):

    def __init__(self, depth= 28, widen_factor= 2, dropRate= 0.0):
        super(WideResNetFeature, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size= 3, stride= 1, padding= 1, bias= False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual= True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt((2. / n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input):
        out = self.conv1(input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return out

class WideResNetClassifier(nn.Module):

    def __init__(self, num_classes, widen_factor= 2):
        super(WideResNetClassifier, self).__init__()
        self.nChannels = 64 * widen_factor
        self.bn1 = nn.BatchNorm2d(self.nChannels, momentum= 0.001)
        self.relu = nn.LeakyReLU(negative_slope= 0.1, inplace= True)
        self.fc = nn.Linear(self.nChannels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt((2. / n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input):
        out = self.relu(self.bn1(input))
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class AlexNetFeature(nn.Module):

    def __init__(self, pret= True):
        super(AlexNetFeature, self).__init__()
        self.feature = models.alexnet(pretrained= pret).features

    def forward(self, input):
        return self.feature(input)

class AlexNetClassifier(nn.Module):

    def __init__(self, num_classes= 345, pret= True):
        super(AlexNetClassifier, self).__init__()
        self.classifier = models.alexnet(pretrained= pret).classifier
        self.classifier[-1] = torch.nn.Linear(4096, num_classes)

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        return self.classifier(input)
