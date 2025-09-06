import pandas as pd
import os
import numpy as np
import glob
import sys

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms
from skimage.transform import resize

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from PIL import Image


class MyLidcDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS, image_size=512):
        """
        IMAGES_PATHS: list of images paths ['./LIDC-IDRI-0001/0001_NI000_slice000.png' ,
                                            './LIDC-IDRI-0001/0001_NI000_slice001.png']

        MASKS_PATHS: list of masks paths ['./LIDC-IDRI-0001/0001_MA000_slice000.png' ,
                                          './LIDC-IDRI-0001/0001_MA000_slice001.png']
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths = MASK_PATHS
        self.image_size = image_size

        self.transformations = transforms.Compose([transforms.ToTensor()])

    def transform(self, image, mask):
        # Transform to tensor

        image = image.convert('L')
        mask = mask.convert('L')

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        image, mask = image.type(
            torch.FloatTensor), mask.type(torch.FloatTensor)

        return image, mask

    def adjust_dimensions(self, image, mask):
        # image resize to the shape

        new_resolution = (self.image_size, self.image_size)
        image = image.resize(new_resolution, Image.Resampling.LANCZOS)
        mask = mask.resize(new_resolution, Image.Resampling.LANCZOS)
        return image, mask

    def __getitem__(self, index):
        cnt_try = 0
        # loop in case if there are any corrupted files
        while cnt_try < 10 and index < self.__len__():
            try:
                image = Image.open(self.image_paths[index])
                mask = Image.open(self.mask_paths[index])

                image, mask = self.adjust_dimensions(image, mask)
                image, mask = self.transform(image, mask)

                return image, mask, self.image_paths[index]
            except Exception as e:
                # if the image is corrupted, load the next image
                print("Corrupted file: ",
                      self.image_paths[index], '  |  ', sys.exc_info()[0])
                print(e)
                index += 1
                cnt_try += 1

        raise ("Could not resolve Corrupted file: ",
               self.image_paths[index], '  |  ', sys.exc_info()[0])

    def __len__(self):
        return len(self.image_paths)


def build_image_path(fname):
    IMAGE_DIR = './CT_Data/Image/'
    patient_id = fname[:4]  # first 4 digits, e.g., '0001'
    folder = f"LIDC-IDRI-{patient_id}"
    return f"{IMAGE_DIR}{folder}/{fname}.png"


def build_mask_path(fname):
    MASK_DIR = './CT_Data/Mask/'
    patient_id = fname[:4]  # first 4 digits
    folder = f"LIDC-IDRI-{patient_id}"
    return f"{MASK_DIR}{folder}/{fname}.png"


def load_dataset(csv_path, augmentation=False, num_out_channels=1, combine_train_val=False, image_size=512, criteria='train'):

    # Meta Information                                                          #
    meta = pd.read_csv(csv_path)
    ############################################################################
    # Get train/test label from meta.csv
    meta['original_image'] = meta['original_image'].apply(build_image_path)
    meta['mask_image'] = meta['mask_image'].apply(build_mask_path)

    train_meta = meta[meta['mode'] == 'train']
    val_meta = meta[meta['mode'] == 'val']

    if criteria == 'test':
        test_meta = meta[meta['mode'] == 'test']
        test_image_paths = list(test_meta['original_image'])
        test_mask_paths = list(test_meta['mask_image'])
        ds = MyLidcDataset(test_image_paths, test_mask_paths, image_size)
        return ds

    # Get all *npy images into list for Train
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])

    # Get all *npy images into list for Validation
    val_image_paths = list(val_meta['original_image'])
    val_mask_paths = list(val_meta['mask_image'])

    if combine_train_val:
        train_image_paths.extend(val_image_paths)
        train_mask_paths.extend(val_mask_paths)

        print("*"*50)
        print("The lenght of image: {}, mask folders: {} for train".format(
            len(train_image_paths), len(train_mask_paths)))
        print("*"*50)

        ds = MyLidcDataset(train_image_paths, train_mask_paths, image_size)
        return ds

    # not combine train and val
    print("*"*50)
    print("The lenght of image: {}, mask folders: {} for train".format(
        len(train_image_paths), len(train_mask_paths)))
    print("The lenght of image: {}, mask folders: {} for validation".format(
        len(val_image_paths), len(val_mask_paths)))
    print("Ratio between Val/ Train is {:2f}".format(
        len(val_image_paths)/len(train_image_paths)))
    print("*"*50)

    # Create Dataset
    train_dataset = MyLidcDataset(
        train_image_paths, train_mask_paths, image_size)
    val_dataset = MyLidcDataset(val_image_paths, val_mask_paths, image_size)
    # test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)

    return train_dataset, val_dataset
