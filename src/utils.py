import csv
import numpy as np

from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import Linear, Sequential, Sigmoid, BCEWithLogitsLoss

from .nets import Net
from .loss_functions import f1_loss, TripletLoss, IncrementalClassRectificationLoss
from .transforms import *
from .datasets import TrainImageDataset, TestImageDataset
from .ddp import partition_dataset


def get_transforms(pretrained=False):
    if pretrained:
        transform = {
            'TRAIN': transforms.Compose(
                            [CombineColors(pretrained),
                             ToPILImage(),
                             RandomResizedCrop(224),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]
            ),
            'DEV': transforms.Compose(
                            [CombineColors(pretrained),
                             ToPILImage(),
                             Resize(256),
                             CenterCrop(224),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]
            )
        }
    else:
        transform = {
            'TRAIN': transforms.Compose(
                            [CombineColors(),
                             NumpyToTensor()
                             ]
            ),
            'DEV': transforms.Compose(
                            [CombineColors(),
                             NumpyToTensor()
                             ]
            )
        }

    return transform

def get_dataset(args, idxs=None, train=True):
    if args.pretrained:
        using_pil = True
    else:
        using_pil = False

    transform = get_transforms(args.pretrained)
    if train:
        image_dir = args.train_images_path
        label_file = args.train_csv_path
        if label_file is None:
            raise ValueError('no label_file provided for training')
        if idxs is None:
            raise ValueError('must specify idxs for training')
        dataset = TrainImageDataset(
                         image_dir=image_dir,
                         label_file=label_file,
                         transform=transform['TRAIN'],
                         idxs=idxs,
                         using_pil=using_pil)
    else:
        image_dir = args.test_images_path
        dataset = TestImageDataset(
                         image_dir=image_dir,
                         transform=transform['DEV'],
                         using_pil=using_pil)

    return dataset

def get_testloader(args, **kwargs):
    testset = get_dataset(args, train=False)
    testloader = DataLoader(testset, shuffle=False, **kwargs)

    return testloader

def get_train_test_split(args, val_split=0.10, distributed=False, **kwargs):
    n_subsample = args.nSubsample

    with open(args.train_csv_path, 'r') as f:
        n_images = sum(1 for row in f.readlines()) - 1 # -1 for header row
    if n_subsample != 0:
        arr = np.random.choice(n_images, n_subsample, replace=False)
        train_idxs = arr[:int(n_subsample * (1 - val_split))]
        dev_idxs = arr[int(n_subsample * (1 - val_split)):]
    else:
        arr = np.random.choice(n_images, n_images, replace=False)
        train_idxs = arr[:int(n_images * (1 - val_split))]
        dev_idxs = arr[int(n_images * (1 - val_split)):]

    trainset = get_dataset(args, idxs=train_idxs)
    devset = get_dataset(args, idxs=dev_idxs)

    if distributed:
        trainLoader, devLoader, args.batchSz = partition_dataset(trainset, devset, args.batchSz)
    else:
        trainLoader = DataLoader(trainset, shuffle=True, **kwargs)
        devLoader = DataLoader(devset, shuffle=False, **kwargs)

    return trainLoader, devLoader

def freeze_pretrained_model_weights(net):
    for param in net.features.parameters():
        param.require_grad = False

    return net

def freeze_pretrained_model_weights_renet(net):
    for param in net.parameters():
        param.requires_grad = False

def swap_last_layer(net):
    num_features = net.classifier[-1].in_features
    features = list(net.classifier.children())[:-1] # Remove last layer
    features.extend([Linear(num_features, 28)]) # Add our layer with 28 outputs. activation in loss function
    net.classifier = Sequential(*features) # Replace the model classifier

    return net

def swap_last_layer_resnet(net):
    num_features = net.fc.in_features
    net.fc = Linear(num_features, 28)

    return net

def get_network(args):
    name = args.network_name
    pretrained = args.pretrained
    if hasattr(models, name):
        torchvision_model = getattr(models, name)
        net = torchvision_model()
        if pretrained:
        # Freeze training for all layers
        # Newly created modules have require_grad=True by default
            if 'resnet' in name:
                freeze_pretrained_model_weights_renet(net)
            else:
                freeze_pretrained_model_weights(net)

        if 'resnet' in name:
            net = swap_last_layer_resnet(net)
        else:
            net = swap_last_layer(net)

    else:
        net = Net()

    return net

def get_loss_function(lf='bce', args=None):
    if lf == 'bce':
        return BCEWithLogitsLoss()

    elif lf == 'f1':
        return f1_loss

    elif lf == 'crl':
        if args:
            return IncrementalClassRectificationLoss(*args)
        raise ValueError('args for CRL not found')
    else:
        raise ModuleNotFoundError('loss function not found')

def positive_predictions(predictions):
    positives = []

    for prediction in predictions:
        output = []
        i = 0
        for label in prediction:
            if(label == 1):
                output.append(str(i))
            i += 1
        positives.append(' '.join(output))

    return positives

def predict(args, net, dataLoader, predF):
    net.eval()

    with torch.no_grad():
        predF.write('Id,Predicted\n')
        print('writing predictions...')
        for batch_idx, data in enumerate(dataLoader):
            inputs, image_ids = data['image'], data['image_id']
            if args.cuda:
                inputs = inputs.cuda()

            outputs = net(inputs)
            if args.sigmoid:
                outputs = torch.sigmoid(outputs)
            if args.thresholds is not None:
                thresholds = [float(val) for val in
                                            args.thresholds.split(",")]

                thresholds = torch.tensor(thresholds)
                if args.cuda:
                    thresholds = thresholds.cuda()
                pred = outputs.data.gt(thresholds)
            else:
                pred = outputs.data.gt(0.5)
            preds = positive_predictions(pred)
            for _ in zip(image_ids, preds):
                predF.write(",".join(_) + '\n')
                predF.flush()
