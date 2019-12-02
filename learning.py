import torch
import os
import numpy as np
import pandas as pd 
import time 
from torchvision.utils import make_grid
from utils import TensorboardWriter, MetricTracker
from torch.autograd import Variable
class Learning(object):
    def __init__(self,
            model,
            criterion,
            optimizer,
            scheduler,
            metric_ftns,
            device,
            num_epoch,
            grad_clipping,
            grad_accumulation_steps,
            early_stopping,
            validation_frequency,
            tensorboard,
            checkpoint_dir,
            resume_path):
        self.device, device_ids = self._prepare_device(device)
        # self.model = model.to(self.device)
        
        self.start_epoch = 1
        if resume_path is not None:
            self._resume_checkpoint(resume_path)
        if len(device_ids) > 1:
            # self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            self.model = torch.nn.DataParallel(model)
            # cudnn.benchmark = True
        self.model = model.cuda()
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.num_epoch = num_epoch 
        self.scheduler = scheduler
        self.grad_clipping = grad_clipping
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping = early_stopping
        self.validation_frequency =validation_frequency
        self.checkpoint_dir = checkpoint_dir
        self.best_epoch = 1
        self.best_score = 0
        self.writer = TensorboardWriter(os.path.join(checkpoint_dir, 'tensorboard'), tensorboard)
        self.train_metrics = MetricTracker('loss', writer = self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer = self.writer)
        
    def train(self, train_dataloader):
        score = 0
        for epoch in range(self.start_epoch, self.num_epoch+1):
            print("{} epoch: \t start training....".format(epoch))
            start = time.time()
            train_result  = self._train_epoch(epoch, train_dataloader)
            train_result.update({'time': time.time()-start})
            
            for key, value in train_result.items():
                print('    {:15s}: {}'.format(str(key), value))

            # if (epoch+1) % self.validation_frequency!=0:
            #     print("skip validation....")
            #     continue
            # print('{} epoch: \t start validation....'.format(epoch))
            # start = time.time()
            # valid_result = self._valid_epoch(epoch, valid_dataloader)
            # valid_result.update({'time': time.time() - start})
            
            # for key, value in valid_result.items():
            #     if 'score' in key:
            #         score = value 
            #     print('   {:15s}: {}'.format(str(key), value))
            score+=0.001
            self.post_processing(score, epoch)
            if epoch - self.best_epoch > self.early_stopping:
                print('WARNING: EARLY STOPPING')
                break
    def _train_epoch(self, epoch, data_loader):
        self.model.train()
        self.optimizer.zero_grad()
        self.train_metrics.reset()
        for idx, (data, target) in enumerate(data_loader):
            data = Variable(data.cuda())
            target = [ann.to(self.device) for ann in target]
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.writer.set_step((epoch - 1) * len(data_loader) + idx)
            self.train_metrics.update('loss', loss.item())
            if (idx+1) % self.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (idx+1) % int(np.sqrt(len(data_loader))) == 0:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        return self.train_metrics.result()
    def _valid_epoch(self, epoch, data_loader):
        self.valid_metrics.reset()
        self.model.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.set_step((epoch - 1) * len(data_loader) + idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        
        return self.valid_metrics.result()
    def post_processing(self, score, epoch):
        best = False
        if score > self.best_score:
            self.best_score = score 
            self.best_epoch = epoch 
            best = True
            print("best model: {} epoch - {:.5}".format(epoch, score))
        self._save_checkpoint(epoch = epoch, save_best = best)
        
        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.get_state_dict(self.model),
            'best_score': self.best_score
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict
    
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    
    @staticmethod
    def _prepare_device(device):
        n_gpu_use = len(device)
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        list_ids = device
        device = torch.device('cuda:{}'.format(device[0]) if n_gpu_use > 0 else 'cpu')
        
        return device, list_ids