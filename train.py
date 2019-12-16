import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.efficientdet import EfficientDet
from models.losses import FocalLoss
from datasets import VOCDetection, COCODetection, CocoDataset, get_augumentation, detection_collate
from utils import EFFICIENTDET


parser = argparse.ArgumentParser(
    description='EfficientDet Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/root/data/VOCdevkit/',
                    help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
parser.add_argument('--network', default='efficientdet-d0', type=str,
                    help='efficientdet-[d0, d1, ..]')

parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_epoch', default=500, type=int,
                    help='Num epoch for training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_worker', default=16, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--num_classes', default=20, type=int,
                    help='Number of class used in model')
parser.add_argument('--device', default=[0, 1], type=list,
                    help='Use CUDA to train model')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='Number of gradient accumulation steps')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./saved/weights/', type=str,
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def prepare_device(device):
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    list_ids = device
    device = torch.device('cuda:{}'.format(
        device[0]) if n_gpu_use > 0 else 'cpu')

    return device, list_ids


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict


checkpoint = []
if(args.resume is not None):
    resume_path = str(args.resume)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(
        args.resume, map_location=lambda storage, loc: storage)
    args.num_classes = checkpoint['num_classes']
    args.network = checkpoint['network']

train_dataset = []
if(args.dataset == 'VOC'):
    train_dataset = VOCDetection(root=args.dataset_root,
                                 transform=get_augumentation(phase='train', width=EFFICIENTDET[args.network]['input_size'], height=EFFICIENTDET[args.network]['input_size']))

elif(args.dataset == 'COCO'):
    train_dataset = CocoDataset(root_dir=args.dataset_root, set_name='train2017', transform=get_augumentation(
        phase='train', width=EFFICIENTDET[args.network]['input_size'], height=EFFICIENTDET[args.network]['input_size']))

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_worker,
                              shuffle=True,
                              collate_fn=detection_collate,
                              pin_memory=True)

model = EfficientDet(num_classes=args.num_classes,
                     network=args.network,
                     W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                     D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                     D_class=EFFICIENTDET[args.network]['D_class'],
                     )
if(args.resume is not None):
    model.load_state_dict(checkpoint['state_dict'])
device, device_ids = prepare_device(args.device)
model = model.to(device)
if(len(device_ids) > 1):
    model = torch.nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, verbose=True)
criterion = FocalLoss()


def train():
    model.train()
    iteration = 1
    for epoch in range(args.num_epoch):
        print("{} epoch: \t start training....".format(epoch))
        start = time.time()
        result = {}
        total_loss = []
        optimizer.zero_grad()
        for idx, (images, annotations) in enumerate(train_dataloader):
            images = images.to(device)
            annotations = annotations.to(device)
            classification, regression, anchors = model(images)
            classification_loss, regression_loss = criterion(
                classification, regression, anchors, annotations)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                print('loss equal zero(0)')
                continue
            loss.backward()
            if (idx+1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

            total_loss.append(loss.item())
            if(iteration % 100 == 0):
                print('{} iteration: training ...'.format(iteration))
                ans = {
                    'epoch': epoch,
                    'iteration': iteration,
                    'cls_loss': classification_loss.item(),
                    'reg_loss': regression_loss.item(),
                    'mean_loss': np.mean(total_loss)
                }
                for key, value in ans.items():
                    print('    {:15s}: {}'.format(str(key), value))

            iteration += 1
        scheduler.step(np.mean(total_loss))
        result = {
            'time': time.time() - start,
            'loss': np.mean(total_loss)
        }
        for key, value in result.items():
            print('    {:15s}: {}'.format(str(key), value))
        arch = type(model).__name__
        state = {
            'arch': arch,
            'num_class': args.num_classes,
            'network': args.network,
            'state_dict': get_state_dict(model)
        }
        torch.save(
            state, './weights/checkpoint_{}_{}_{}.pth'.format(args.dataset, args.network, epoch))
    state = {
        'arch': arch,
        'num_class': args.num_class,
        'network': args.network,
        'state_dict': get_state_dict(model)
    }
    torch.save(state, './weights/Final_{}.pth'.format(args.network))


if __name__ == '__main__':
    train()
