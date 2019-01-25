import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging


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

    tp = torch.sum(input * target, dim=0) # size = [1, ncol]
    tn = torch.sum((1 - target) * (1 - input), dim=0) # size = [1, ncol]
    fp = torch.sum((1 - target) * input, dim=0) # size = [1, ncol]
    fn = torch.sum(target * (1 - input), dim=0) # size = [1, ncol]
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    # f1 = where(f1 != f1, zeros(f1.size()), f1)
    return 1 - f1.mean()

def get_minority_classes(y, batchSz):
    sorted_hjk, ix = y.sum(0).sort()
    mask = torch.cumsum(sorted_hjk, 0) <= .5 * batchSz
    sorted_hjk = sorted_hjk[mask]
    sorted_, sorted_ix = ix[mask].sort()

    return sorted_[sorted_hjk[sorted_ix] > 1]

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
        distance_positive = F.l1_loss(anchor, positive, reduction='sum')
        distance_negative = F.l1_loss(anchor, negative, reduction='sum')
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class IncrementalClassRectificationLoss(nn.Module):

    def __init__(self,
        margin,
        alpha,
        batchSz,
        k,
        class_level_hard_mining=True,
        sigmoid=True
    ):
        super(IncrementalClassRectificationLoss, self).__init__()

        self.margin = margin
        self.alpha = alpha
        self.batchSz = batchSz
        self.k = k
        self.class_level_hard_mining = class_level_hard_mining
        self.sigmoid = sigmoid
        self.trip_loss = TripletLoss(margin)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target, X):
        bce = self.bce(input, target)
        idxs = get_minority_classes(target, batchSz=self.batchSz)
        if self.sigmoid:
            input = torch.sigmoid(input)
            y_min = target[:, idxs]
            preds_min = input[:, idxs]
        else:
            y_min = target[:, idxs]
            preds_min = input[:, idxs]

        y_mask = y_min == 1
        P = torch.nonzero(y_mask)
        N = torch.nonzero(~y_mask)
        preds_P = preds_min[y_mask]

        k = self.k
        idx_tensors = []
        pred_tensors = []
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
            # create 2d array with indices for anchor, pos and neg examples
            grid = torch.stack(torch.meshgrid([torch.Tensor(a).long(), torch.arange(n_p), torch.arange(n_n)])).reshape(3, -1).t()
            idx_tensors.append(torch.cat([P[grid[:, 0]], pos_idxs[grid[:, 1]], neg_idxs[grid[:, 2]]], 1))
            pred_tensors.append(torch.stack([preds_P[grid[:, 0]], pos_preds[grid[:, 1]], neg_preds[grid[:, 2]]], 1))

        try:
            if self.class_level_hard_mining:
                idx_tensors = torch.cat(idx_tensors, 0)
                pred_tensors = torch.cat(pred_tensors, 0)
            else:
                # TODO: implement instance level hard mining
                pass
            crl = self.trip_loss(pred_tensors[:, 0], pred_tensors[:, 1], pred_tensors[:, 2])
            loss = self.alpha * crl + (1 - self.alpha) * bce

            return loss

        except RuntimeError:
            # Getting runtime error in torch.cat above as sometimets there are
            # no index or pred tensors to combine from hard mining
            logging.warning('RuntimeError in loss statement')

            return bce
