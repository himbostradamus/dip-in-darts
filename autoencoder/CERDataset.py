import torch
from torch.utils.data import Dataset
import fnmatch
import os
import numpy as np


# Definition of the Dataset for Training-/Test data
# image: pytorch tensor of size 1 x 128 x 128
# landmarks: = pytorch tensor of size 8


class CERDataset(Dataset):
    """CER dataset"""

    def __init__(self, landmark_dir, image_dir):
        """
        Args:
            landmark_dir (string): Directory with all the landmarks.
            image_dir (string): Directory with all the images.
        """
        self.landmark_dir = landmark_dir
        self.image_dir = image_dir

    def __len__(self):
        return len(fnmatch.filter(os.listdir(self.image_dir), '*.pt*'))

    def __getitem__(self, idx):
        landmarks = torch.load(self.landmark_dir + '/labels_' + str(idx + 1) + '.pt')
        image = torch.load(self.image_dir + '/image_' + str(idx + 1) + '.pt')
        sample = {'image': image, 'landmarks': landmarks}
        return sample
