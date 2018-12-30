import os
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from random import Random
from torch.autograd import Variable


class DistributedDataParallel(nn.Module):
    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        self.first_call = True

        def allreduce_params():
            if (self.needs_reduction):
                self.needs_reduction = False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)

            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def weight_broadcast(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0)

    def forward(self, *inputs, **kwargs):
        if self.first_call:
            print("first broadcast start")
            self.weight_broadcast()
            self.first_call = False
            print("first broadcast done")
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(train_set, val_set, gbatch_size):
    """ Partitioning dataset """
    size = dist.get_world_size()
    batch_size = int(gbatch_size / float(size))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=(val_sampler is None),
                                             sampler=val_sampler)
    return train_loader, val_loader, batch_size


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def init_print(rank: int, size: int, debug_print: bool = True):
    """
    Creates new stdout to display rank and size for distributed training for
    simpler/clearer monitoring.

    Args:
        rank (int): rank of machine in distributed process
        size (int): total number of machines in distributed process
        debug_print (bool): If False, mutes stdout from all nodes except
            master. Default is True.
    """
    if not debug_print:
        if rank > 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    else:
        old_out = sys.stdout

        class LabeledStdout:
            def __init__(self, rank, size):
                self._r = rank
                self._s = size
                self.flush = sys.stdout.flush

            def write(self, x):
                if x == '\n':
                    old_out.write(x)
                else:
                    old_out.write('[%d/%d] %s' % (self._r, self._s, x))

        sys.stdout = LabeledStdout(rank, size)
