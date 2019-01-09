import torch

# base libraries
import argparse
import os
import setproctitle
import shutil
import csv
import datetime
import pytz
from google.cloud import storage

# internals
from src import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET'] if('CLOUD_STORAGE_BUCKET' in os.environ) else ""
BRANCH_NAME = os.environ['BRANCH'] if('BRANCH' in os.environ) else "probably-master"

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
    parser.add_argument('--thresholds', type=str, default=None)
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

    net = get_network(args)
    if args.load:
        print("Loading network: {}".format(args.load))
        load_model(args, net)
    else:
        load_path = 'work/%s/%s' % (args.network_name, args.dataset_name)
        files = [f for f in os.listdir(load_path) if \
                            os.path.isfile(os.path.join(load_path, f)) \
                            and '.pth' in f]
        current = max([int(i.replace('.pth', '')) for i in files])
        args.load = os.path.join(load_path, str(current) + '.pth')
        load_model(args, net)

    if args.cuda:
        net = net.cuda()

    now = datetime.datetime.now(tz=pytz.timezone("US/Mountain")).strftime("%Y-%m-%d___%H:%M:%S")
    predict_csv_path = os.path.join(args.save, '{}_{}_predict.csv'.format(BRANCH_NAME, now))

    predF = open(predict_csv_path, 'a')

    predict(args, net, testLoader, predF)

    predF.close

    if len(CLOUD_STORAGE_BUCKET) != 0:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
        blob = bucket.blob(predict_csv_path)

        blob.upload_from_filename(predict_csv_path)

def load_model(args, net):
    load_path = args.load
    net.load_state_dict(torch.load(load_path))

if __name__ == '__main__':
    main()
