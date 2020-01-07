import yaml
import numpy as np
import torch
import os
import requests
import socket
import datetime
import json


def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def init_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict
