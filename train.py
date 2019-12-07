from data import *
from utils.augmentations import SSDAugmentation
from models.efficientdet import EfficientDet
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from torchvision import transforms

from models.losses import FocalLoss

from datasets import VOCDetection, get_augumentation, detection_collate


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/root/data/VOCdevkit/',
                    help='Dataset root directory path')
parser.add_argument('--num_epoch', default=500, type=int,
                    help='Batch size for training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=12, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--num_class', default=21, type=int,
                    help='Number of class used in model')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict

def train():
    train_dataset = VOCDetection(root = args.dataset_root,
                        transform= get_augumentation(phase='train'))
    train_dataloader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    model = EfficientDet(num_classes=args.num_class)
    if(torch.cuda.is_available()):
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model = model.cuda()
    if(args.resume is not None):
        state = torch.load(args.resume, map_location=lambda storage, loc: storage)
        state_dict = state['state_dict']
        num_class = state['num_class']
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    criterion = FocalLoss()
    model.train()
    iteration = 0
    
    for epoch in range(args.num_epoch):
        print('Start epoch: {} ...'.format(epoch))
        total_loss = []
        for idx, (images, annotations) in enumerate(train_dataloader):
            images = images.cuda()
            annotations = annotations.cuda()
            classification, regression, anchors = model(images)
            classification_loss, regression_loss = criterion(classification, regression, anchors, annotations)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss
            if bool(loss == 0):
                print('loss equal zero(0)')
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss.append(loss.item())

            if(iteration%100==0):
                print('Epoch/Iteration: {}/{}, classification: {}, regression: {}, totol_loss: {}'.format(epoch, iteration, classification_loss.item(), regression_loss.item(), np.mean(total_loss)))
            iteration+=1
        scheduler.step(np.mean(total_loss))
        arch = type(model).__name__
        state = {
            'arch': arch,
            'num_class': args.num_class,
            'state_dict': get_state_dict(model)
        }
        torch.save(state, './weights/checkpoint_{}.pth'.format(epoch))
    model.eval()
    state = {
        'arch': arch,
        'num_class': args.num_class,
        'state_dict': get_state_dict(model)
    }
    torch.save(state, './weights/FinalWeights.pth')

if __name__ == '__main__':
    train()
