import logging
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
           'vgg19', 'vgg19_bn']

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicLinear(nn.Module):

    def __init__(self, in_feature, out_feature, classification):
        super(BasicLinear, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=True)
        self.classification = classification

    def forward(self, x):
        x = self.linear(x)
        if self.classification:
            return torch.sigmoid(x)
        else:
            return torch.tanh(x)

class VGG(nn.Module):
    '''
        VGG Model
    '''

    def __init__(self, features, classification, w=256, h=256, n_classes=1):
        super(VGG, self).__init__()

        self.logger = logging.getLogger(__name__)

        _w = int(w/(2**5))
        _h = int(h/(2**5))
        self.features = features
        modules = [nn.Dropout(),
            nn.Linear(_w*_h*128, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, n_classes)]
        if classification:
            modules.append(nn.LogSoftmax(dim=1))
        # else:
            # modules.append(nn.Tanh())

        self.classifier = nn.Sequential(*modules)

        self.logger.info(self.classifier)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    'A': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'], #vgg11
    'B': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'], #vgg13
    'D': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'], #vgg16
    'E': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', # vgg19
          128, 128, 128, 128, 'M'],
}

def vgg11(w=256, h=256, n_classes=1):
    """VGG 11-layer model (configuration "A")"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['A']), classification, w=w, h=h, n_classes=n_classes)

def vgg11_bn(w=256, h=256, n_classes=1):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['A'], batch_norm=True), classification, w=w, h=h, n_classes=n_classes)


def vgg13(w=256, h=256, n_classes=1):
    """VGG 13-layer model (configuration "B")"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['B']), classification, w=w, h=h, n_classes=n_classes)


def vgg13_bn(w=256, h=256, n_classes=1):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['B'], batch_norm=True), classification, w=w, h=h, n_classes=n_classes)


def vgg16(w=256, h=256, n_classes=1):
    """VGG 16-layer model (configuration "D")"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['D']), classification, w=w, h=h, n_classes=n_classes)


def vgg16_bn(w=256, h=256, n_classes=1):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['D'], batch_norm=True), classification, w=w, h=h, n_classes=n_classes)


def vgg19(w=256, h=256, n_classes=1):
    """VGG 19-layer model (configuration "E")"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['E']), classification, w=w, h=h, n_classes=n_classes)


def vgg19_bn(w=256, h=256, n_classes=1):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    classification = True if (n_classes == 2 or n_classes == 3) else False
    return VGG(make_layers(cfg['E'], batch_norm=True), classification, w=w, h=h, n_classes=n_classes)