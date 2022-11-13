import torch
import torch.nn as nn
import torch.nn.functional as F


######### Backbone models #########
#### VGG-13
class BlockVGG(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockVGG, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        return out


class VGG(nn.Module):
    def __init__(self, block, num_classes=10, cfg=None):
        super(VGG, self).__init__()
        self.cfg = cfg 

        self.layers = self._make_layers(in_planes=3, block=block)
        self.output_dim = self.cfg[-1]
        self.linear = nn.Linear(self.cfg[-1] if isinstance(self.cfg[-1], int) else self.cfg[-1][0], num_classes)

    def _make_layers(self, in_planes, block):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, use_linear=True):
        out = self.layers(x)
        out = out.mean(dim=(2,3))
        if(use_linear):
            out = self.linear(out)
        return out

def VGGmodel(num_classes=10):
    cfg = [64, (64, 2), 128, (128, 2), 256, (256, 2), 512, (512, 2), 512, 512]
    return VGG(BlockVGG, num_classes=num_classes, cfg=cfg)


#### ResNets
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out) 


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out) 


# Model class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.output_dim = 512*block.expansion
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_linear=True):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(use_linear):
            out = self.linear(out)
        return out

class ResNet_basic(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):
        super(ResNet_basic, self).__init__()

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.output_dim = 512*block.expansion
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_linear=True):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(use_linear):
            out = self.linear(out)
        return out


def get_block(block):
    if(block=="BasicBlock"):
        return BasicBlock
    elif(block=="Bottleneck"):
        return Bottleneck

def ResNet18(num_classes=10, block="BasicBlock"):
    return ResNet(get_block(block), [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10, block="BasicBlock"):
    return ResNet(get_block(block), [3,4,6,3], num_classes=num_classes)

def ResNet56(num_classes=10, block="BasicBlock"):
    return ResNet_basic(get_block(block), [9,9,9], num_classes=num_classes)


### Retrieval function for backbones ###
def create_model(name, num_classes=10, block='BasicBlock'):
    if(name == 'VGG-13'):
        net = VGGmodel(num_classes=num_classes)
    elif(name == 'res18'):
        net = ResNet18(num_classes=num_classes, block=block)

    return net