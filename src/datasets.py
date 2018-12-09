import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from skimage import io, img_as_float # transform

def to_one_hot(df):
    tmp = df.Target.str.get_dummies(sep=' ')
    tmp.columns = map(int, tmp.columns)
    return df.join(tmp.sort_index(axis=1))

def get_image_ids_from_dir_contents(image_dir):
    all_images = [name for name in os.listdir(image_dir) \
                  if os.path.isfile(os.path.join(image_dir, name))]
    return list(set([name.split('_')[0] for name in all_images]))

class TrainImageDataset(Dataset):
    """Fluorescence microscopy images of protein structures training dataset"""

    def __init__(self,
        image_dir,
        label_file,
        transform=None,
        idxs=None,
        using_pil=False
    ):
        """
        Args:
            label_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.idxs = idxs
        self.labels = to_one_hot(pd.read_csv(label_file))
        self.using_pil = using_pil
        if self.idxs is not None:
            self.labels = self.labels.iloc[self.idxs, :].\
                                                reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_red = img_name + '_red.png'
        img_blue = img_name + '_blue.png'
        img_green = img_name + '_green.png'
        img_yellow = img_name + '_yellow.png'

        if self.using_pil:
            pth2img = lambda x: io.imread(x)
        else:
            pth2img = lambda x: img_as_float(io.imread(x))

        img_red = pth2img(os.path.join(self.image_dir, img_red))
        img_blue = pth2img(os.path.join(self.image_dir, img_blue))
        img_green = pth2img(os.path.join(self.image_dir, img_green))
        img_yellow = pth2img(os.path.join(self.image_dir, img_yellow))
        labels = self.labels.iloc[idx, 2:].values
        labels = labels.astype('int')
        sample = {'image_id': img_name,
                  'image_red': img_red,
                  'image_blue': img_blue,
                  'image_green': img_green,
                  'image_yellow': img_yellow,
                  'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

class TestImageDataset(Dataset):
    """Fluorescence microscopy images of protein structures test dataset"""

    def __init__(self,
        image_dir,
        transform=None,
        using_pil=False
    ):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_ids = get_image_ids_from_dir_contents(image_dir)
        self.image_dir = image_dir
        self.transform = transform
        self.using_pil = using_pil

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_red = img_name + '_red.png'
        img_blue = img_name + '_blue.png'
        img_green = img_name + '_green.png'
        img_yellow = img_name + '_yellow.png'

        if self.using_pil:
            pth2img = lambda x: io.imread(x)
        else:
            pth2img = lambda x: img_as_float(io.imread(x))

        img_red = pth2img(os.path.join(self.image_dir, img_red))
        img_blue = pth2img(os.path.join(self.image_dir, img_blue))
        img_green = pth2img(os.path.join(self.image_dir, img_green))
        img_yellow = pth2img(os.path.join(self.image_dir, img_yellow))
        sample = {'image_id': img_name,
                  'image_red': img_red,
                  'image_blue': img_blue,
                  'image_green': img_green,
                  'image_yellow': img_yellow,
                  'labels' : np.zeros(28)}

        if self.transform:
            sample = self.transform(sample)

        return sample
