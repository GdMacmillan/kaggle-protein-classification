import torchvision.models as models
import torch.nn as nn

from torch import cat


RESNET_ENCODERS = {
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152,
}

VGG_CLASSIFIERS = {
    11: models.vgg11,
    13: models.vgg13,
    16: models.vgg16,
    19: models.vgg19,
}

VGG_BN_CLASSIFIERS = {
    11: models.vgg11_bn,
    13: models.vgg13_bn,
    16: models.vgg16_bn,
    19: models.vgg19_bn,
}


class Net(nn.Module):
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

        if pretrained:
            for param in encoder.parameters():
                param.requires_grad=False

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1.weight = nn.Parameter(cat((w,w[:,:1,:,:]),dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        num_features = encoder.fc.in_features
        self.fc = nn.Linear(num_features, 28)

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
            vgg_net = VGG_BN_CLASSIFIERS[n_layers](pretrained=pretrained)
        else:
            vgg_net = VGG_CLASSIFIERS[n_layers](pretrained=pretrained)

        if pretrained:
            for param in vgg_net.features.parameters():
                param.requires_grad=False
            for param in vgg_net.classifier.parameters():
                param.requires_grad=False

        # initialize conv2d to take in 4 channels instead of 3
        feature_layers = []
        w = vgg_net.features[0].weight
        conv2d = nn.Conv2d(4, 64, kernel_size=3, padding=1) # Create 2d conv layer
        conv2d.weight = nn.Parameter(cat((w,w[:,:1,:,:]),dim=1))
        feature_layers.append(conv2d)

        remaining_features = list(vgg_net.features.children())[1:] # Remove first layer
        feature_layers.extend(remaining_features)

        # swap last layer for fc layer with 28 outputs
        num_features = vgg_net.classifier[-1].in_features
        classifier_layers = list(vgg_net.classifier.children())[:-1] # Remove last layer
        classifier_layers.extend([nn.Linear(num_features, 28)]) # Add layer with 28 outputs. activation in loss function

        self.features = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet34(pretrained):
    net = Resnet4Channel(encoder_depth=34, pretrained=pretrained)
    return net

def resnet50(pretrained):
    net = Resnet4Channel(encoder_depth=50, pretrained=pretrained)
    return net

def resnet101(pretrained):
    net =  Resnet4Channel(encoder_depth=101, pretrained=pretrained)
    return net

def resnet152(pretrained):
    net = Resnet4Channel(encoder_depth=152, pretrained=pretrained)
    return net

def vgg11(pretrained):
    net = VGG4Channel(n_layers=11, batch_norm=False, pretrained=pretrained)
    return net

def vgg13(pretrained):
    net = VGG4Channel(n_layers=13, batch_norm=False, pretrained=pretrained)
    return net

def vgg16(pretrained):
    net = VGG4Channel(n_layers=16, batch_norm=False, pretrained=pretrained)
    return net

def vgg19(pretrained):
    net = VGG4Channel(n_layers=19, batch_norm=False, pretrained=pretrained)
    return net

def vgg11_bn(pretrained):
    net = VGG4Channel(n_layers=11, batch_norm=True, pretrained=pretrained)
    return net

def vgg13_bn(pretrained):
    net = VGG4Channel(n_layers=13, batch_norm=True, pretrained=pretrained)
    return net

def vgg16_bn(pretrained):
    net = VGG4Channel(n_layers=16, batch_norm=True, pretrained=pretrained)
    return net

def vgg19_bn(pretrained):
    net = VGG4Channel(n_layers=19, batch_norm=True, pretrained=pretrained)
    return net

def baseline(pretrained):
    if pretrained:
        print('Baseline net not pretrained. Training from scratch')
    return Net()
