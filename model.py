import pooling
from config import CFG

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

import transformers
import tokenizers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
os.environ['TOKENIZERS_PARALLELISM']='true'


def re_initializing_layer(model, config, layer_num):
    for module in model.model.encoder.layer[-layer_num:].modules():
        if isinstance(module, nn.Linear):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data) 
                
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
                
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return model   


class FB3Model(nn.Module):
    def __init__(self, CFG, config_path = None, pretrained = False):
        super().__init__()
        self.CFG = CFG
        if config_path is None:
            self.config = AutoConfig.from_pretrained(CFG.model_dir, ouput_hidden_states=True)
            self.config.save_pretrained(CFG.OUTPUT_DIR + 'config')
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        self.config.output_hidden_states = True
            
        CFG.LOGGER.info(self.config)
        
        if pretrained:
            self.model = AutoModel.from_pretrained(CFG.model_dir, config=self.config)
            self.model.save_pretrained(CFG.OUTPUT_DIR + 'model')
        else:
            self.model = AutoModel(self.config)
            
        if self.CFG.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if CFG.pooling == 'mean':
            self.pool = pooling.MeanPooling(self.config.hidden_size)
        elif CFG.pooling == 'max':
            self.pool = pooling.MaxPooling(self.config.hidden_size)
        elif CFG.pooling == 'min':
            self.pool = pooling.MinPooling(self.config.hidden_size)
        elif CFG.pooling == 'attention':
            self.pool = pooling.AttentionPooling(self.config.hidden_size)
        elif CFG.pooling == 'cls':
            self.pool = pooling.ClsPooling(self.config.hidden_size)
        elif CFG.pooling == 'mean_max':
            self.pool = pooling.MeanMaxPooling(self.config.hidden_size)
        elif CFG.pooling == 'conv1d':
            self.pool = pooling.Conv1dPooling(self.config.hidden_size)
        else:
            if CFG.pooling == 'weightedlayer':
                pooling_class = pooling.WeightedLayerPooling
            elif CFG.pooling == 'attn_layer_wise':
                pooling_class = pooling.LayerWiseAttnPooling
            elif CFG.pooling == 'layerwise_cls':
                pooling_class = pooling.LayerWiseCLSPooling
            elif CFG.pooling == 'concat':
                pooling_class = pooling.ConcatPooling
            elif CFG.pooling == 'lstm':
                pooling_class = pooling.LSTMPooling
            elif CFG.pooling == 'gru':
                pooling_class = pooling.GRUPooling
            else:
                raise NotImplementedError 
            if CFG.decouple_pooling is False:
                self.pool = pooling_class(self.config.hidden_size, CFG.n_layer) 
            else:
                self.pool = pooling.DecoupledPooling(pooling_class, self.config.hidden_size, CFG.n_layer) 
        
        # self.fc = nn.Linear(self.config.hidden_size, self.CFG.n_targets)
        # self._init_weights(self.fc)
        
        if 'deberta-v2-xxlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False)
        if 'deberta-v2-xlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False)
        if 'funnel-transformer-xlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.blocks[:1].requires_grad_(False)
        if 'funnel-transformer-large' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.blocks[:1].requires_grad_(False)
        if 'deberta-large' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:16].requires_grad_(False)
        if 'deberta-xlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:36].requires_grad_(False)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
                
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
                
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def feature(self, inputs):
        outputs = self.model(**inputs)
        if CFG.pooling in ['mean', 'max', 'min', 'attention', 'cls', 'mean_max', 'conv1d']:
            last_hidden_states = outputs.last_hidden_state
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
        else:
            all_layer_embeddings = torch.stack(outputs.hidden_states)[-CFG.n_layer:]
            attention_mask = inputs['attention_mask'].unsqueeze(0).unsqueeze(-1).float()
            # use mean pooling for each layer
            sum_embeddings = torch.sum(all_layer_embeddings * attention_mask, 2)
            sum_mask = attention_mask.sum(2)
            sum_mask = torch.clamp(sum_mask, min = 1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            feature = self.pool(mean_embeddings)
            
        return feature
    
    def forward(self, inputs):
        out = self.feature(inputs)
        # outout = self.fc(out)
        return out