import numpy as np
import torch

from torchvision import transforms


class CombineColors(object):
    """Combines the the image in a sample to a given size."""

    def __call__(self, sample):
        img_name = sample['image_id']
        img_red = sample['image_red']
        img_blue = sample['image_blue']
        img_green = sample['image_green']
        img_yellow = sample['image_yellow']
        labels = sample['labels']
        image = np.dstack((img_red, img_green, img_blue, img_yellow))

        return {'image': image, 'labels': labels, 'image_id': img_name}


class ToPILImage(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.ToPILImage(self.mode)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class RandomResizedCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.RandomResizedCrop(self.size)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.RandomHorizontalFlip()(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class Resize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, size=256):
        self.size = size

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.Resize(self.size)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class CenterCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.CenterCrop(self.size)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.ToTensor()(image)

        return {'image': image.type(torch.FloatTensor),
                'labels': torch. \
                    from_numpy(labels).type(torch.FloatTensor),
                'image_id': img_name}


class NumpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch. \
                    from_numpy(image).type(torch.FloatTensor),
                'labels': torch. \
                    from_numpy(labels).type(torch.FloatTensor),
                'image_id': img_name}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.Normalize(self.mean, self.std)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.\
                                            format(self.mean, self.std)
