from tqdm import tqdm
import argparse
import os
import random
import shutil
from collections import OrderedDict
import time
import warnings
import epdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_warmup as warmup

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from models.efficientdet import EfficientDet
from models.losses import FocalLoss
from datasets import VOCDetection, CocoDataset, get_augumentation, detection_collate, Resizer, Normalizer, Augmenter, collater
from utils import EFFICIENTDET, get_state_dict
from eval import evaluate, evaluate_coco


breakpoint = epdb.set_trace

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument(
    '--dataset_root',
    default='/root/data/VOCdevkit/',
    help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
parser.add_argument('--network', default='efficientdet-d0', type=str,
                    help='efficientdet-[d0, d1, ..]')

parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_epoch', default=500, type=int,
                    help='Num epoch for training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_class', default=20, type=int,
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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')
parser.add_argument('--eval_epochs', default=5, type=int,
                    help='after how many training epochs will do evaluation (default 5).')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='freeze EfficientNet-d{x} backbone')
parser.add_argument('--freeze_bn', action='store_true',
                    help='freeze all batch norm layers')
parser.add_argument('--mixed_training', action='store_true',
                    help='Use AMP mixed training optimization O1')

iteration = 1


def train(train_loader, model, scheduler, warmup_scheduler, optimizer, epoch, args):
    global iteration
    print("{} epoch: \t start training....".format(epoch))
    start = time.time()
    total_loss = []
    model.train()
    model.module.is_training = True
    optimizer.zero_grad()
    for idx, (images, annotations) in tqdm(enumerate(train_loader),
                                           total=len(train_loader)):
        images = images.float().cuda()
        annotations = annotations.cuda()
        classification_loss, regression_loss = model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss
        if bool(loss == 0):
            print('loss equal zero(0)')
            continue

        if args.mixed_training:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (idx + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            # if warmup_scheduler:
            #     warmup_scheduler.dampen()

        total_loss.append(loss.item())
        if(iteration % 10 == 0):
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
    scheduler.step(np.mean(total_loss))  # used for ReduceLROnPlateau

    result = {
        'time': time.time() - start,
        'loss': np.mean(total_loss)
    }
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))


def test(dataset, model, epoch, args):
    print("{} epoch: \t start validation....".format(epoch))
    model = model.module
    model.eval()
    model.is_training = False
    with torch.no_grad():
        if(args.dataset == 'VOC'):
            evaluate(dataset, model)
        else:
            evaluate_coco(dataset, model)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            # args.rank = int(os.environ["RANK"])
            args.rank = 1
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    # Training dataset
    train_dataset = []
    if(args.dataset == 'VOC'):
        train_dataset = VOCDetection(root=args.dataset_root, transform=transforms.Compose(
            [Normalizer(), Augmenter(), Resizer()]))
        valid_dataset = VOCDetection(root=args.dataset_root, image_sets=[(
            '2007', 'test')], transform=transforms.Compose([Normalizer(), Resizer()]))
        args.num_class = train_dataset.num_classes()
    elif(args.dataset == 'COCO'):
        train_dataset = CocoDataset(
            root_dir=args.dataset_root,
            set_name='train2017',
            transform=transforms.Compose(
                [
                    Normalizer(),
                    Augmenter(),
                    Resizer()]))
        valid_dataset = CocoDataset(
            root_dir=args.dataset_root,
            set_name='val2017',
            transform=transforms.Compose(
                [
                    Normalizer(),
                    Resizer()]))
        args.num_class = train_dataset.num_classes()

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)

    checkpoint = []
    if(args.resume is not None):
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        args.start_epoch = checkpoint['epoch'] + 1
        del params

    model = EfficientDet(num_classes=args.num_class,
                         network=args.network,
                         W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                         D_class=EFFICIENTDET[args.network]['D_class']
                         )
    if(args.resume is not None):
        tmp = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            k = k.replace("module.", "")
            tmp[k] = v
        model.load_state_dict(tmp)
        del tmp

    model.to("cuda")

    if args.freeze_backbone:
        model.freeze_backbone()

    if args.freeze_bn:
        model.freeze_bn()

    # define loss function (criterion) , optimizer, scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr)
    if args.resume is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    num_steps = len(train_loader) * args.num_epoch

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=num_steps)
    if args.resume is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if args.mixed_training:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level="O1",
                                          keep_batchnorm_fp32=None,
                                          loss_scale=128)

    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    warmup_scheduler = None

    if args.resume is not None and "warmup_scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["warmup_scheduler"])
    del checkpoint

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            print('Run with DistributedDataParallel with divice_ids....')
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Run with DistributedDataParallel without device_ids....')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()
        print('Run with DataParallel ....')
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.num_epoch):
        train(train_loader, model, scheduler, warmup_scheduler,
              optimizer, epoch, args)

        state = {
            'epoch': epoch,
            'parser': args,
            'state_dict': get_state_dict(model),
            'optimizer': optimizer.state_dict()
        }

        torch.save(
            state,
            os.path.join(
                args.save_folder,
                args.dataset,
                args.network,
                "checkpoint_{}.pth".format(epoch)))

        if (epoch + 1) % args.eval_epochs == 0:
            test(valid_dataset, model, epoch, args)


def main():
    args = parser.parse_args()
    if(not os.path.exists(os.path.join(args.save_folder, args.dataset, args.network))):
        os.makedirs(os.path.join(args.save_folder, args.dataset, args.network))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '2'
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == "__main__":
    main()
