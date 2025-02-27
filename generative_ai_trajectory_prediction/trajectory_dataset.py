import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class TrajectoryDataset(Dataset):
    """
    Custom Dataset for pairing consecutive images.
    Each sample is a tuple: (current_image, next_image)
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Get list of image files sorted by filename
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(self.image_files) < 2:
            raise ValueError("Not enough images to create pairs.")

    def __len__(self):
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        current_image_path = self.image_files[idx]
        next_image_path = self.image_files[idx + 1]

        current_image = Image.open(current_image_path).convert('RGB')
        next_image = Image.open(next_image_path).convert('RGB')

        if self.transform:
            current_image = self.transform(current_image)
            next_image = self.transform(next_image)

        return current_image, next_image
