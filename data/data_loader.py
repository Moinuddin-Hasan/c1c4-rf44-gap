# ======================================================================
# This file handles all data loading and data augmentation tasks.
# It defines the Albumentations transforms for training and testing,
# downloads the CIFAR-10 dataset, and creates the PyTorch DataLoaders.
# ======================================================================

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader

# Pre-computed mean and std for CIFAR-10 for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

class AlbumentationsWrapper:
    """A wrapper to make Albumentations transforms compatible with torchvision datasets."""
    def __init__(self, transforms): 
        self.transforms = transforms
    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']

def get_loaders(batch_size=512):
    """
    Configures and returns train and test DataLoaders with specified augmentations.
    """
    # Training augmentations as specified in the assignment
    train_tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=1, max_height=16, max_width=16,
            min_holes=1, min_height=16, min_width=16,
            fill_value=CIFAR10_MEAN,
            p=0.5
        ),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])
    
    # Test augmentations only include normalization
    test_tfms = A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = datasets.CIFAR10('./data', train=True,  download=True, transform=AlbumentationsWrapper(train_tfms))
    test_dataset  = datasets.CIFAR10('./data', train=False, download=True, transform=AlbumentationsWrapper(test_tfms))
    
    # Create DataLoaders
    loader_args = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    
    return train_loader, test_loader