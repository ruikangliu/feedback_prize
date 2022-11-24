import os
import gc
import re
import ast
import sys
import copy
import json
import time
import datetime
import math
import string
import pickle
import random
import joblib
import itertools
from distutils.util import strtobool
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint


class CFG:
    str_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train = True
    debug = False
    offline = False
    models_path = 'FB3-models'
    epochs = 5
    save_all_models = False
    competition = 'FB3'
    apex = True
    print_freq = 20
    num_workers = 4
    # model name
    """
        microsoft/deberta-v3-base
        deberta-v2-xxlarge
        deberta-v2-xlarge
        funnel-transformer-xlarge
        funnel-transformer-large
        deberta-large
        deberta-xlarge
    """
    model = 'microsoft/deberta-v3-base' #If you want to train on the kaggle platform, v3-base is realistic. v3-large will time out.
    loss_func = 'RMSE' # 'SmoothL1', 'RMSE'
    gradient_checkpointing = True
    scheduler = 'cosine'
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    #Layer-Wise Learning Rate Decay
    llrd = True
    layerwise_lr = 5e-5
    layerwise_lr_decay = 0.9
    layerwise_weight_decay = 0.01
    layerwise_adam_epsilon = 1e-6
    layerwise_use_bertadam = False
    #pooling
    """
        Last Hidden Output:
            mean: Mean Pooling
            max: Max Pooling
            min: Min Pooling
            attention: Attention Pooling
            cls: [CLS] Embed + FC
            mean_max: Mean + Max Pooling
            conv1d: Conv1D Pooling (slow)
            
        
        Hidden States Output:
            weightedlayer: Weighted Layer Pooling
            attn_layer_wise: Layer-wise Attention Pooling
            layerwise_cls: Layerwise CLS Embeddings
            concat: Concatenate Pooling
            lstm: LSTM pooling (slow)
            gru: GRU pooling (slow)
    """ 
    pooling = 'mean'
    n_layer = 4
    #init_weight
    init_weight = 'normal' # normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal
    #re-init
    reinit = True
    reinit_n = 1
    #adversarial
    fgm = False
    awp = False
    adv_lr = 1
    adv_eps = 0.2
    unscale = True
    eps = 1e-6
    betas = (0.9, 0.999)
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000    # grad clip
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    cv_seed = 42
    n_fold = 5
    trn_fold = list(range(n_fold))
    batch_size = 4
    n_targets = 6
    gpu_id = 0
    
    train_file = './dataset/train.csv'
    test_file = './dataset/test.csv'
    submission_file = './dataset/sample_submission.csv'
    

def config_setup():
    CFG.device = f'cuda:{CFG.gpu_id}'
    CFG.model_dir = os.path.join('/home/ys/lrk/modelzoo/transformers', CFG.model)
    #Unique model name
    if len(CFG.model.split("/")) == 2:
        # {CFG.str_now}
        CFG.identifier = f'{CFG.model.split("/")[1]}'
    else:
        CFG.identifier = f'{CFG.model}'
    CFG.identifier += f'/{CFG.pooling}_{CFG.loss_func}'
    print(CFG.identifier)

    # Read train and split with MultilabelStratifiedKFold
    if CFG.train:
        CFG.df_train = pd.read_csv(CFG.train_file)
        CFG.OUTPUT_DIR = f'./output/{CFG.identifier}/'
        CFG.log_filename = CFG.OUTPUT_DIR + 'train'
        if CFG.offline:
            #TO DO
            pass
        else:
            os.system('pip install iterative-stratification==0.1.7')
        #CV
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold    
        Fold = MultilabelStratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.cv_seed)
        for n, (train_index, val_index) in enumerate(Fold.split(CFG.df_train, CFG.df_train[CFG.target_cols])):
            CFG.df_train.loc[val_index, 'fold'] = int(n)
        CFG.df_train['fold'] = CFG.df_train['fold'].astype(int)
    else:
        #TO DO
        pass

    if CFG.debug:
        CFG.epochs = 2
        CFG.trn_fold = [0]
        if CFG.train:
            CFG.df_train = CFG.df_train.sample(n = 100, random_state = CFG.seed).reset_index(drop=True)
            
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)    
    print(CFG.OUTPUT_DIR)  