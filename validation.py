import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Module.losses import BCEDiceLoss
from Module.dataset import load_dataset
from Module.metrics import iou_score, dice_coef, dice_coef2, calculate_hd95
from Module.utils import AverageMeter, str2bool

from Module.Unet.unet_model import UNet
from Module.UnetNested.Nested_Unet import NestedUNet
from Module.ELUnet.elunet import ELUnet


def save_output(title, output, output_dir):
    to_pil = transforms.ToPILImage()

    title = title.replace("NI", "PD")
    output_directory = os.path.join(output_dir, title)

    pil_image_1 = to_pil(output.squeeze().cpu())
    pil_image_1 = pil_image_1.convert('L')

    pil_image_1.save(output_directory)


def main():
    augmentations = False
    model_name = 'UNET'  # choices: 'UNET', 'NestedUNET', 'ELUnet'
    epochs = 400
    batch_size = 2
    early_stopping = 50
    num_workers = 0
    optimizer_name = 'Adam'
    learning_rate = 1e-5
    momentum_rate = 0.9
    weight_decay_rate = 1e-4
    nesterov = False
    image_size = 256

    print('-' * 20)
    print("Configuration Setting as follow")
    print('augmentations: ', augmentations)
    print('model_name: ', model_name)
    print('epochs: ', epochs)
    print('batch_size: ', batch_size)
    print('early_stopping: ', early_stopping)
    print('num_workers: ', num_workers)
    print('optimizer: ', optimizer_name)
    print('lr: ', learning_rate)
    print('momentum: ', momentum_rate)
    print('weight_decay: ', weight_decay_rate)
    print('nesterov: ', nesterov)
    print('image_size: ', image_size)
    print('-' * 20)

    if augmentations == True:
        NAME = model_name + '_with_augmentation'
    else:
        NAME = model_name + '_base'

    print("=> creating model")
    if model_name == 'UNET':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    elif model_name == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    elif model_name == 'ELUnet':
        model = ELUnet(in_channels=1, out_channels=1)
    else:
        raise NotImplementedError(model_name + ' is not implemented!')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print("Loading model file from {}".format(NAME))

    checkpoint = torch.load('model_outputs/{}/best.pth'.format(NAME))
    print('Epoch: ', checkpoint['epoch'])
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    OUTPUT_MASK_DIR = 'Segmentation_output/{}'.format(NAME)
    print("Saving OUTPUT files in directory {}".format(OUTPUT_MASK_DIR))
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    test_dataset = load_dataset(
        csv_path='./CT_Data/split_dataset.csv',
        image_size=image_size,
        criteria='test',
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    avg_meters = {'iou': AverageMeter(), 'dice': AverageMeter()}

    dp = {'title': [], 'dice_score': [], 'ioU': [], 'hd95': []}
    df = pd.DataFrame(dp)

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target, path in test_loader:
            input = input.cuda()
            target = target.cuda()
            title = path[0].split('/')[-1]

            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef2(output, target)
            hd95 = calculate_hd95(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])

            new_row = pd.DataFrame([[title, dice, iou, hd95]], columns=[
                                   'title', 'dice_score', 'ioU', 'hd95'])
            df = pd.concat([df, new_row], axis=0)

            save_output(title, output, OUTPUT_MASK_DIR)

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        df.to_csv(os.path.join(OUTPUT_MASK_DIR, 'results.csv'), index=False)
        print("Results saved at {}".format(
            os.path.join(OUTPUT_MASK_DIR, 'results.csv')))


if __name__ == '__main__':
    main()
