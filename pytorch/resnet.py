import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Bottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,   64,  256,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256,  128, 512,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512,  256, 1024, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, 512, 2048, num_blocks[3], stride=2) # [batch, 2048, 1, 1]
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc5 = nn.Linear(2048, num_class)

    def forward(self, x):   # x: [batch, 3, 224, 224]
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc5(out)
        return out

    def _make_layer(self, block, in_channel, mid_channel, out_channel, num_block, stride):
        layers = []
        layers.append(block(in_channel, mid_channel, out_channel, stride))
        for i in range(num_block-1):
            layers.append(block(out_channel, mid_channel, out_channel, stride=1))
        return nn.Sequential(*layers)


def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

if __name__ == '__main__':
    from torch.autograd import Variable
    net = ResNet50()
    y = net(Variable(torch.randn(1,3,224,224)))
    print(y.size())
