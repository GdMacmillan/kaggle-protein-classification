import torch

# base libraries
import argparse
import os
import setproctitle
import shutil
import csv
from google.cloud.storage import Client

# internals
from src import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET'] if('CLOUD_STORAGE_BUCKET' in os.environ) else ""
GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE']

default_train_images = os.path.join(BASE_DIR, 'data/train_images')
default_csv = os.path.join(BASE_DIR, 'data/train.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)
    parser.add_argument('-p', '--pretrained', type=bool, default=False)
    parser.add_argument('-dp', '--data-parallel', type=bool, default=True)
    parser.add_argument('--train-images-path', type=str, default=default_train_images)
    parser.add_argument('--train-csv-path', type=str, default=default_csv)
    parser.add_argument('-l', '--load',
                    help='if using load, must be path to .pth file containing serialized model state dict')
    parser.add_argument('--batchSz', type=int, default=32) # 64
    parser.add_argument('--nEpochs', type=int, default=1) # 300
    parser.add_argument('--sEpoch', type=int, default=1)
    parser.add_argument('--unfreeze-epoch', type=int, default=-1)
    parser.add_argument('--nSubsample', type=int, default=0)
    parser.add_argument('--use-cuda', type=str, default='no')
    parser.add_argument('--nGPU', type=int, default=0)
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--crit', type=str, default='bce', choices=('bce', 'f1', 'crl'))
    parser.add_argument('--distributed', type=bool, default=False,
                    help='If True, use distributed data parallel training (default, False).')
    args = parser.parse_args()

    if args.use_cuda == 'yes' and not torch.cuda.is_available():
        raise ValueError('Use cuda requires cuda devices and ' + \
                         'drivers to be installed. Please make ' + \
                         'sure both are installed.'
                        )
    elif args.use_cuda == 'yes' and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False

    if args.cuda and args.nGPU == 0:
        nGPU = 1
    else:
        nGPU = args.nGPU



    print('testing upload to google cloud storage bucket')
    if len(CLOUD_STORAGE_BUCKET) != 0:
        storage_client = Client.from_service_account_json(
        GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE)
        print('client authenticated')
        bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
        all_files = [name for name in os.listdir(args.save) \
                            if os.path.isfile(os.path.join(args.save, name))]
        if len(all_files) > 0:
            print("uploading weights...")
            for file in all_files:
                blob = bucket.blob(os.path.join(args.save, file))
                blob.upload_from_filename(os.path.join(args.save, file))








    main_proc = True
    if args.distributed:
        dist.init_process_group(backend='gloo')
        main_proc = dist.get_rank() == 0
        init_print(dist.get_rank(), dist.get_world_size())

    print("using cuda ", args.cuda)

    args.save = args.save or 'work/%s/%s' % \
                                (args.network_name, args.dataset_name)
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save, exist_ok=True)

    kwargs = {'batch_size': args.batchSz}

    trainLoader, devLoader = get_train_test_split(args, **kwargs)

    # net = get_network(args)
    net = NETWORKS_DICT[args.network_name](args.pretrained)

    if args.load:
        print("Loading network: {}".format(args.load))
        load_model(args, net)

    if args.distributed:
        net = DistributedDataParallel(net)
    elif args.data_parallel:
        net = torch.nn.DataParallel(net)

    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), weight_decay=1e-4)
    else:
        raise ModuleNotFoundError('optimiser not found')

    if args.crit == 'crl':
        lf_args = [0.5, 8.537058595265812e-06, args.batchSz, 5, True, True]
    else:
        lf_args = None

    criterion = get_loss_function(args.crit, lf_args)

    sched_args = [10, 1e-4, 1.1, .5, -1]
    scheduler = CosineAnnealingRestartsLR(optimizer, *sched_args)

    trainF = open(os.path.join(args.save, 'train.csv'), 'a')
    testF = open(os.path.join(args.save, 'test.csv'), 'a')

    for epoch in range(args.sEpoch, args.nEpochs + args.sEpoch):
        # adjust_opt(args, epoch, optimizer)
        scheduler.step()
        unfreeze_weights(args, epoch, net)
        train(args, epoch, net, trainLoader, criterion, optimizer, trainF)
        test(args, epoch, net, devLoader, criterion, optimizer, testF)
        if main_proc:
            save_model(args, epoch, net)

    trainF.close()
    testF.close()

    # if len(CLOUD_STORAGE_BUCKET) != 0:
    #     storage_client = Client.from_service_account_json(
    #     GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE)
    #     bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    #     all_files = [name for name in os.listdir(args.save) \
    #                         if os.path.isfile(os.path.join(args.save, name))]
    #     if len(all_files) > 0:
    #         print("uploading weights...")
    #         for file in all_files:
    #             blob = bucket.blob(os.path.join(args.save, file))
    #             blob.upload_from_filename(os.path.join(args.save, file))

def train(args, epoch, net, trainLoader, criterion, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, data in enumerate(trainLoader):
        inputs, labels = data['image'], data['labels']
        # get the inputs
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        if args.crit == 'crl':
            loss_inputs = (outputs, labels, inputs)
        else:
            loss_inputs = (outputs, labels)

        loss = criterion(*loss_inputs)
        loss.backward()
        if args.distributed:
            average_gradients(net)
        optimizer.step()
        nProcessed += len(data)
        pred = outputs.data.gt(0.5)
        tp = (pred + labels.data.byte()).eq(2).sum().float()
        fp = (pred - labels.data.byte()).eq(1).sum().float()
        fn = (pred - labels.data.byte()).eq(255).sum().float()
        tn = (pred + labels.data.byte()).eq(0).sum().float()
        acc = (tp + tn) / (tp + tn + fp + fn)
        try:
            prec = tp / (tp + fp)
        except ZeroDivisionError:
            prec = 0.0
        try:
            rec = tp / (tp + fn)
        except ZeroDivisionError:
            rec = 0.0
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\tPrec: {:.4f}\tRec: {:.4f}\tTP: {}\tFP: {}\tFN: {}\tTN: {}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), acc, prec, rec, tp, fp, fn, tn))
        trainF.write('{},{},{},{},{}\n'.format(partialEpoch, loss.item(), acc, prec, rec))
        trainF.flush()

def test(args, epoch, net, devLoader, criterion, optimizer, testF):
    net.eval()
    test_loss = 0
    acc = prec = rec = 0
    incorrect = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(devLoader):
            inputs, labels = data['image'], data['labels']
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = net(inputs)

            if args.crit == 'crl':
                loss_inputs = (outputs, labels, inputs)
            else:
                loss_inputs = (outputs, labels)

            test_loss += criterion(*loss_inputs)
            pred = outputs.data.gt(0.5)
            tp = (pred + labels.data.byte()).eq(2).sum().float()
            fp = (pred - labels.data.byte()).eq(1).sum().float()
            fn = (pred - labels.data.byte()).eq(255).sum().float()
            tn = (pred + labels.data.byte()).eq(0).sum().float()
            acc += (tp + tn) / (tp + tn + fp + fn)
            try:
                prec += tp / (tp + fp)
            except ZeroDivisionError:
                prec += 0.0
            try:
                rec += tp / (tp + fn)
            except ZeroDivisionError:
                rec += 0.0
        test_loss /= len(devLoader)
        acc /= len(devLoader)
        prec /= len(devLoader)
        rec /= len(devLoader)
        print('\nTest set: Loss: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}\n'.format(
            test_loss, acc, prec, rec))
        testF.write('{},{},{},{},{}\n'.format(epoch, test_loss, acc, prec, rec))
        testF.flush()

def save_model(args, epoch, net):
    save_path = os.path.join(args.save, '%d.pth' % epoch)
    net = net.module if args.distributed or args.data_parallel else net
    torch.save(net.state_dict(), save_path)

def load_model(args, net):
    load_path = args.load
    if args.cuda:
        net.load_state_dict(torch.load(load_path))
    else:
        net.load_state_dict(torch.load(load_path, map_location='cpu'))

def adjust_opt(args, epoch, optimizer):
    if args.opt == 'sgd':
        if epoch < 15: lr = 1e-3
        elif epoch == 18: lr = 5e-4
        elif epoch == 20: lr = 1e-4
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def unfreeze_weights(args, epoch, net):
    if args.unfreeze_epoch == -1 or epoch < args.unfreeze_epoch:
        pass
    else:
        net = net.module if args.distributed or args.data_parallel else net
        if 'resnet' in args.network_name:
            for param in net.parameters():
                param.require_grad = True
            print('params unfrozen')
        else:
            for param in net.features.parameters():
                param.require_grad = True
            for param in net.classifier.parameters():
                param.require_grad = True

if __name__ == '__main__':
    main()
