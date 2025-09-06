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

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Module.losses import BCEDiceLoss
from Module.dataset import load_dataset
from Module.metrics import iou_score, dice_coef
from Module.utils import AverageMeter, str2bool

from Module.Unet.unet_model import UNet
from Module.UnetNested.Nested_Unet import NestedUNet
from Module.ELUnet.elunet import ELUnet


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    augmentations = False
    model_name = 'UNET'  # choices: 'UNET', 'NestedUNET', 'ELUnet'
    epochs = 400
    batch_size = 1
    early_stopping = 50
    num_workers = 0
    optimizer_name = 'Adam'
    learning_rate = 1e-5
    momentum_rate = 0.9
    weight_decay_rate = 1e-4
    nesterov = False
    image_size = 256

    if augmentations == True:
        file_name = model_name + '_with_augmentation'
    else:
        file_name = model_name + '_base'
    os.makedirs('model_outputs/{}'.format(file_name), exist_ok=True)
    print("Creating directory called", file_name)

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

    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model")
    if model_name == 'UNET':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    elif model_name == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    elif model_name == 'ELUnet':
        model = ELUnet(in_channels=1, out_channels=1)
    else:
        raise NotImplementedError(model_name + ' is not implemented!')

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(
            params, lr=learning_rate, weight_decay=weight_decay_rate)
    elif weight_decay_rate == 'SGD':
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum_rate,
                              nesterov=nesterov, weight_decay=weight_decay_rate)
    else:
        raise NotImplementedError

    train_dataset, val_dataset = load_dataset(
        csv_path='./CT_Data/split_dataset.csv',
        image_size=image_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    log = pd.DataFrame([], columns=['epoch', 'lr', 'loss',
                                    'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])

    best_dice = 0
    trigger = 0
    epoch = 0

    if os.path.isfile('model_outputs/{}/last_model.pth'.format(file_name)):
        print("=> loading checkpoint")
        checkpoint = torch.load(
            'model_outputs/{}/last_model.pth'.format(file_name))

        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        trigger = checkpoint['trigger']
        best_dice = checkpoint['best_dice']
        optimizer.load_state_dict(checkpoint['optimizer'])

        # load the csv
        prev_log = pd.read_csv('model_outputs/{}/log.csv'.format(file_name))
        log = pd.concat([prev_log, log], ignore_index=True)

        print("=> loaded checkpoint")
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print("=> loaded checkpoint (best_dice {})".format(
            checkpoint['best_dice']))
        print("=> loaded checkpoint (trigger {})".format(
            checkpoint['trigger']))

    while epoch < epochs:

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
            epoch + 1, epochs, train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'], val_log['iou']))

        tmp = pd.DataFrame([[
            epoch,
            learning_rate,
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice']]], columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])

        log = pd.concat([log, tmp], ignore_index=True)

        log.to_csv('model_outputs/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        torch.save({
            'epoch': epoch,
            'best_dice': best_dice,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'trigger': trigger,
        }, 'model_outputs/{}/last_model.pth'.format(file_name))

        if val_log['dice'] > best_dice:
            torch.save({
                'epoch': epoch,
                'best_dice': best_dice,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'trigger': trigger,
            }, 'model_outputs/{}/best.pth'.format(file_name))

            best_dice = val_log['dice']
            print(
                "=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if early_stopping >= 0 and trigger >= early_stopping:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
        epoch += 1


if __name__ == '__main__':
    main()
