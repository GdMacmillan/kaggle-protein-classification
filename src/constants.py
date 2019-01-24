from .nets import resnet34, resnet50, resnet101, resnet152, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

NETWORKS_DICT = {
    'resnet34' : resnet34,
    'resnet50' : resnet50,
    'resnet101' : resnet101,
    'resnet152' : resnet152,
    'vgg11' : vgg11,
    'vgg13' : vgg13,
    'vgg16' : vgg16,
    'vgg19' : vgg19,
    'vgg11_bn' : vgg11_bn,
    'vgg13_bn' : vgg13_bn,
    'vgg16_bn' : vgg16_bn,
    'vgg19_bn' : vgg19_bn,
}
