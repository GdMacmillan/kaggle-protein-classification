import torch
import torchvision

from torch import nn


RESNET_ENCODERS = {
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}

VGG_CLASSIFIERS = {
    11: torchvision.models.vgg11,
    13: torchvision.models.vgg13,
    16: torchvision.models.vgg16,
    19: torchvision.models.vgg19,
}

VGG_BN_CLASSIFIERS = {
    11: torchvision.models.vgg11_bn,
    13: torchvision.models.vgg13_bn,
    16: torchvision.models.vgg16_bn,
    19: torchvision.models.vgg19_bn,
}

def freeze_pretrained_model_weights(net, type='vgg'):
    if type='vgg'
        for param in net.features.parameters()[1:-1]:
            param.require_grad = False
    else:
        for param in net.parameters()[1:-1]:
            param.requires_grad = False

def swap_last_layer(net, type='vgg'):
    num_features = net.classifier[-1].in_features
    features = list(net.classifier.children())[:-1] # Remove last layer
    features.extend([Linear(num_features, 28)]) # Add our layer with 28 outputs. activation in loss function
    net.classifier = Sequential(*features) # Replace the model classifier

    return net

def swap_last_layer_resnet(net):
    num_features = net.fc.in_features
    net.fc = Linear(num_features, 28)

    return net


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5) # 4 channel in, 6 channels out, filter size 5
        self.pool = nn.MaxPool2d(2, 2) # 6 channel in, 6 channels out, filter size 2, stride 2
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 channel in, 16 channels out, filter size 5
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(x))
        x = x.view(-1, 16 * 125 * 125)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return x


class Resnet4Channel(nn.Module):
    def __init__(self, encoder_depth=34, pretrained=True, num_classes=28):
        super().__init__()

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.conv1.weight
        w.requires_grad = False if pretrained else True
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.conv1.weight = nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        self.fc = nn.Linear(512 * (1 if encoder_depth==34 else 4), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class VGG4Channel(nn.Module):
    def __init__(self, n_layers=11, batch_norm=False, pretrained=True, num_classes=28):
        super().__init__()

        if batch_norm:
            classifier = VGG_BN_CLASSIFIERS[n_layers](pretrained=pretrained)
        else:
            classifier = VGG_CLASSIFIERS[n_layers](pretrained=pretrained)

        w = classifier.features[0].weight
        w.requires_grad = False if pretrained else True
        self.conv2d = nn.Conv2d(4, 64, kernel_size=3, padding=1)

        features = list(net.features.children())[1:]







def resnet34(pretrained):
    net = Resnet4Channel(encoder_depth=34)
    if pretrained:
        freeze_pretrained_model_weights(net, 'resnet')
    return net

def resnet50(pretrained):
    net = Resnet4Channel(encoder_depth=50)
    if pretrained:
        freeze_pretrained_model_weights(net, 'resnet')
    return net

def resnet101(pretrained):
    net =  Resnet4Channel(encoder_depth=101)
    if pretrained:
        freeze_pretrained_model_weights(net, 'resnet')
    return net

def resnet152(pretrained):
    net = Resnet4Channel(encoder_depth=152)
    if pretrained:
        freeze_pretrained_model_weights(net, 'resnet')
    return net

def vgg11(pretrained):
    net = VGG4Channel(n_layers=11, batch_norm=False)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg13(pretrained):
    net = VGG4Channel(n_layers=13, batch_norm=False)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg16(pretrained):
    net = VGG4Channel(n_layers=16, batch_norm=False)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg19(pretrained):
    net = VGG4Channel(n_layers=19, batch_norm=False)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg11_bn(pretrained):
    net = VGG4Channel(n_layers=11, batch_norm=True)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg13_bn(pretrained):
    net = VGG4Channel(n_layers=13, batch_norm=True)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg16_bn(pretrained):
    net = VGG4Channel(n_layers=16, batch_norm=True)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def vgg19_bn(pretrained):
    net = VGG4Channel(n_layers=19, batch_norm=True)
    if pretrained:
        freeze_pretrained_model_weights(net, 'vgg')
    return net

def baseline(pretrained):
    if pretrained:
        print('Baseline net not pretrained. Training from scratch')
    return Net()
