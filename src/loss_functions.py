from torch import sum, cumsum, where, zeros

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropy_with_logits(input, target):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp()\
                 + (-input - max_val).exp()).log()

    return loss.mean()

def f1_loss(input, target, epsilon=1E-8):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    tp = sum(input * target, dim=0) # size = [1, ncol]
    tn = sum((1 - target) * (1 - input), dim=0) # size = [1, ncol]
    fp = sum((1 - target) * input, dim=0) # size = [1, ncol]
    fn = sum(target * (1 - input), dim=0) # size = [1, ncol]
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    # f1 = where(f1 != f1, zeros(f1.size()), f1)
    return 1 - f1.mean()

def get_minority_classes(y, num_classes):
    ix = np.argsort(y.sum(0))
    sorted_hjk = y.sum(0)[ix]
    mask = cumsum(sorted_hjk, 0) <= .5 * num_classes
    sorted_hjk = sorted_hjk[mask]
    ix = ix[mask]

    return ix[np.argsort(ix)][sorted_hjk[np.argsort(ix)] > 1]


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample

    source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.l1_loss(anchor, positive, size_average=False)
        distance_negative = F.l1_loss(anchor, negative, size_average=False)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class IncrementalClassRectificationLoss(nn.Module):

    def __init__(self, margin, num_classes, k):
        super(IncrementalClassRectificationLoss, self).__init__()

        self.margin = margin
        self.num_classes = num_classes
        self.k = k
        self.triplet_loss = TripletLoss(margin)

    def forward(self, input, target, X):
        idxs = get_minority_classes(target, num_classes=self.num_classes)

        y_min = target[:, idxs]
        preds_min = preds[:, idxs]
        y_mask = y_min == 1
        P = torch.nonzero(y_mask)
        N = torch.nonzero(~y_mask)
        probs_P = preds_min[y_mask]

        k = self.k
        # would like to vectorize this
        for idx, row in enumerate(P):
            anchor_idx, anchor_class = row
            mask = (P[:, 1] == anchor_class)
            mask[idx] = 0
            pos_idxs = P[mask]
            pos_preds, sorted_= preds_min[pos_idxs[:, 0], pos_idxs[:, 1]].sort()
            pos_idxs = pos_idxs[sorted_][:k]
            pos_preds = pos_preds[:k]

            mask = (N[:, 1] == anchor_class)
            neg_idxs = N[mask]
            neg_preds, sorted_= preds_min[neg_idxs[:, 0], neg_idxs[:, 1]].sort()
            neg_idxs = neg_idxs[sorted_][-k:]
            neg_preds = neg_preds[:k]

            a = [idx] # anchor index in P
            n_p = pos_idxs.shape[0]
            n_n = neg_idxs.shape[0]
            grid = torch.stack(torch.meshgrid([torch.arange(0).new_tensor(a), torch.arange(n_p), torch.arange(n_n)])).reshape(-1,3).t()
        #     print(torch.cat([P[grid[:, 0]], pos_idxs[grid[:, 1]], neg_idxs[grid[:, 2]]], 1))
        #     print("")
        #     print(torch.stack([probs_P[grid[:, 0]], pos_preds[grid[:, 1]], neg_preds[grid[:, 2]]], 1))
        #     print("")
