import os 
import argparse 
import collections
from operator import getitem
from functools import reduce
import time 
from pathlib import Path
from utils import load_yaml, init_seed
import importlib
import torch
import pandas as pd 

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from learning import Learning

def split_dataset(file_name):
    df = pd.read_csv(os.path.join(file_name))
    train_df, valid_df = train_test_split(df, random_state=42, stratify=df['label'], test_size=0.2)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    return train_df, valid_df

def getattribute(config, name_package, *args, **kwargs):
    module = importlib.import_module(config[name_package]['PY'])
    module_class = getattr(module, config[name_package]['CLASS'])
    module_args = dict(config[name_package]['ARGS']) if config[name_package]['ARGS'] is not None else dict()
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    package = module_class(*args, **module_args)
    return package

def config_parser(parser, options):
    def _get_opt_name(flags):
        for flg in flags:
            if flg.startswith('--'):
                return flg.replace('--', '')
        return flags[0].replace('--', '')
    def _get_by_path(tree, keys):
        """Access a nested object in tree by sequence of keys."""
        return reduce(getitem, keys, tree)
    def _set_by_path(tree, keys, value):
        """Set a value in a nested object in tree by sequence of keys."""
        keys = keys.split(',')
        _get_by_path(tree, keys[:-1])[keys[-1]] = value
    for opt in options:
        parser.add_argument(*opt.flags, default=None, type=opt.type)
    if not isinstance(parser, tuple):
        args = parser.parse_args()
    config_folder = Path(args.train_cfg.strip("/"))
    config = load_yaml(config_folder)
    if args.device is not None:
        config['DEVICE'] = args.device
    if args.resume is not None:
        config['RESUME_PATH'] = args.resume
    modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
    for key, value in modification.items():
        if value is not None:
            for key in key.split(';'):
                _set_by_path(config, key, value)
            
    return config
    


def main():
    parser = argparse.ArgumentParser(description='Pytorch parser')
    parser.add_argument('--train_cfg', type=str, default='./configs/train_config.yaml', help='train config path')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='OPTIMIZER,ARGS,lr'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='TRAIN_DATALOADER,ARGS,batch_size;VALID_DATALOADER,ARGS,batch_size')
    ]
    config = config_parser(parser, options)
    init_seed(config['SEED'])

    train_df, valid_df = split_dataset(config['DATA_TRAIN'])
    train_dataset = getattribute(config = config, name_package = 'TRAIN_DATASET', df = train_df)

    # from pytoan.pyplot import imshow
    # import matplotlib.pyplot as plt 
    # images = [train_dataset[i][0].permute(1, 2, 0) for i in range(100, 105)]
    # imshow(*images)
    # plt.show()
    
    valid_dataset = getattribute(config = config, name_package = 'VALID_DATASET', df = valid_df)
    train_dataloader = getattribute(config = config, name_package = 'TRAIN_DATALOADER', dataset = train_dataset)
    valid_dataloader = getattribute(config = config, name_package = 'VALID_DATALOADER', dataset = valid_dataset)
    model = getattribute(config = config, name_package = 'MODEL')
    criterion = getattribute(config = config, name_package = 'CRITERION')
    optimizer = getattribute(config = config, name_package= 'OPTIMIZER', params = model.parameters())
    scheduler = getattribute(config = config, name_package = 'SCHEDULER', optimizer = optimizer)
    device = config['DEVICE']
    metric_ftns = [accuracy_score]
    num_epoch = config['NUM_EPOCH']
    gradient_clipping = config['GRADIENT_CLIPPING']
    gradient_accumulation_steps = config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = config['EARLY_STOPPING']
    validation_frequency = config['VALIDATION_FREQUENCY']
    tensorboard = config['TENSORBOARD']
    checkpoint_dir = Path(config['CHECKPOINT_DIR'], type(model).__name__)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    resume_path = config['RESUME_PATH']
    learning = Learning(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler = scheduler,
                        metric_ftns=metric_ftns,
                        device=device,
                        num_epoch=num_epoch,
                        grad_clipping = gradient_clipping,
                        grad_accumulation_steps = gradient_accumulation_steps,
                        early_stopping = early_stopping,
                        validation_frequency = validation_frequency,
                        tensorboard = tensorboard,
                        checkpoint_dir = checkpoint_dir,
                        resume_path=resume_path)
    
    learning.train(tqdm(train_dataloader), tqdm(valid_dataloader))

if __name__ == "__main__":
    main()
