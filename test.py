import torch

# base libraries
import argparse
import os
import setproctitle
import shutil
import csv

# internals
from src import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

default_test_images = os.path.join(BASE_DIR, 'data/test_images')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)
    parser.add_argument('-p', '--pretrained', type=bool, default=False)
    parser.add_argument('-dp', '--data-parallel', type=bool, default=True)
    parser.add_argument('--test-images-path', type=str, default=default_test_images)
    parser.add_argument('-l', '--load')
    parser.add_argument('--batchSz', type=int, default=32)
    parser.add_argument('--save')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    print("using cuda: ", args.cuda)

    args.save = args.save or 'work/%s/%s' % \
                                (args.network_name, args.dataset_name)
    setproctitle.setproctitle('work/%s/%s-test' % \
                                (args.network_name, args.dataset_name))

    if not os.path.exists(args.save):
        raise ValueError('save directory not found')

    kwargs = {'batch_size': args.batchSz}

    testLoader = get_testloader(args, **kwargs)

    if args.load:
        print("Loading network: {}".format(args.load))
        net = torch.load(args.load)
    else:
        load_path = 'work/%s/%s' % (args.network_name, args.dataset_name)
        files = [f for f in os.listdir(load_path) if \
                            os.path.isfile(os.path.join(load_path, f)) \
                            and '.pth' in f]
        current = max([int(i.replace('.pth', '')) for i in files])
        if args.cuda:
            net = torch.load(os.path.join(load_path, str(current) + '.pth'))
        else:
            net = torch.load(os.path.join(load_path, str(current) + '.pth'), map_location='cpu')

    predF = open(os.path.join(args.save, 'predict.csv'), 'a')

    predict(args, net, testLoader, predF)

    predF.close

if __name__ == '__main__':
    main()
