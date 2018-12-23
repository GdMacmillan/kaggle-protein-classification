import csv
import numpy as np

from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import Linear, Sequential, Sigmoid, BCEWithLogitsLoss

from .nets import Net
from .loss_functions import f1_loss
from .transforms import *
from .datasets import TrainImageDataset, TestImageDataset

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

def get_train_test_split(args, val_split=0.10, **kwargs):
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

    trainloader = DataLoader(trainset, shuffle=True, **kwargs)
    devloader = DataLoader(devset, shuffle=False, **kwargs)

    return trainloader, devloader

def get_network(network_name, pretrained=False, lf='bce'):
    if network_name not in ['vgg16']:
        pretrained = False # no pretrained weights for non torchvision models

    if network_name == 'vgg16':
        vgg16 = models.vgg16(pretrained)
        print(vgg16.classifier[6].out_features) # 1000

        # Freeze training for all layers
        # Newly created modules have require_grad=True by default
        if pretrained:
            for param in vgg16.features.parameters():
                param.require_grad = False

        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1] # Remove last layer
        if lf == 'bce':
            features.extend([Linear(num_features, 28)]) # Add our layer with 28 outputs. activation in loss function
        else:
            features.extend([Linear(num_features, 28), Sigmoid()])
        vgg16.classifier = Sequential(*features) # Replace the model classifier

        return vgg16

    else:
        net = Net()

        return net

def get_loss_function(lf='bce'):
    if lf == 'bce':

        return BCEWithLogitsLoss()

    elif lf == 'f1':

        return f1_loss

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
        predF.write('Id, Predicted\n')
        for batch_idx, data in enumerate(dataLoader):
            inputs, image_ids = data['image'], data['image_id']
            if args.cuda:
                inputs = inputs.cuda()

            outputs = net(inputs)
            pred = outputs.data.gt(0.5)
            preds = positive_predictions(pred)
            for _ in zip(image_ids, preds):
                predF.write(",".join(_) + '\n')
                predF.flush()