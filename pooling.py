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


class Pooling(nn.Module):
    def _init_weights(self, module):
        self.config = AutoConfig.from_pretrained(CFG.model_dir, ouput_hidden_states=True)
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

"""
    Last Hidden Output
"""
class MeanPooling(Pooling):
    def __init__(self, hidden_size, use_fc=True):
        super(MeanPooling, self).__init__()
        self.use_fc = use_fc
        self.fc = nn.Linear(hidden_size, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return self.fc(mean_embeddings) if self.use_fc is True else mean_embeddings


class MeanAugPooling(Pooling):
    def __init__(self, hidden_size, use_fc=True):
        super(MeanPooling, self).__init__()
        self.use_fc = use_fc
        self.fc = nn.Linear(hidden_size, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return self.fc(mean_embeddings) if self.use_fc is True else mean_embeddings


class MaxPooling(Pooling):
    def __init__(self, hidden_size, use_fc=True):
        super(MaxPooling, self).__init__()
        self.use_fc = use_fc
        self.fc = nn.Linear(hidden_size, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return self.fc(max_embeddings) if self.use_fc is True else max_embeddings
   
    
class MinPooling(Pooling):
    def __init__(self, hidden_size):
        super(MinPooling, self).__init__()
        self.fc = nn.Linear(hidden_size, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return self.fc(min_embeddings)


#Attention pooling
class AttentionPooling(Pooling):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.fc = nn.Linear(hidden_size, CFG.n_targets)
        self._init_weights(self.fc)

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return self.fc(attention_embeddings)


class ClsPooling(Pooling):
    def __init__(self, hidden_size):
        super(ClsPooling, self).__init__()
        self.fc = nn.Linear(hidden_size, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        return self.fc(last_hidden_state[:, 0])


class MeanMaxPooling(Pooling):
    def __init__(self, hidden_size):
        super(MeanMaxPooling, self).__init__()
        self.mean_pooling = MeanPooling(hidden_size, use_fc=False)
        self.max_pooling = MaxPooling(hidden_size, use_fc=False)
        self.fc = nn.Linear(hidden_size * 2, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        mean_embed = self.mean_pooling(last_hidden_state, attention_mask)
        max_embed = self.max_pooling(last_hidden_state, attention_mask)
        embed = torch.cat([mean_embed, max_embed], 1)
        mean_max_embed = self.fc(embed)
        return mean_max_embed


class Conv1dPooling(Pooling):
    def __init__(self, hidden_size):
        super(Conv1dPooling, self).__init__()
        self.cnn1 = nn.Conv1d(hidden_size, 128, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(128, 64, kernel_size=2, padding=0)
        self.fc = nn.Linear(64, CFG.n_targets)
        self._init_weights(self.fc)
        
    def forward(self, last_hidden_state, attention_mask):
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
        cnn_embeddings = self.cnn2(cnn_embeddings).permute(0, 2, 1)
        cnn_embeddings[attention_mask == 0] = -1e4
        max_embeddings, _ = torch.max(cnn_embeddings, dim=1)
        return self.fc(max_embeddings)


#------------------------------------------------
"""
    Hidden States Output
"""
class WeightedLayerPooling(Pooling):
    def __init__(self, hidden_size, n_layer=4, n_target=CFG.n_targets):
        super(WeightedLayerPooling, self).__init__()
        self.n_layer = n_layer
        self.layer_weights = nn.Parameter(
                torch.tensor([1] * (n_layer), dtype=torch.float)
            )
        self.fc = nn.Linear(hidden_size, n_target)
        self._init_weights(self.fc)

    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1)
        weighted_average = (weight_factor*layer_wise_embed).sum(dim=0) / self.layer_weights.sum()

        return self.fc(weighted_average)
   
    
class LayerWiseAttnPooling(Pooling):
    def __init__(self, hidden_size, n_layer=4, n_target=CFG.n_targets, hiddendim_fc=128):
        super(LayerWiseAttnPooling, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(CFG.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(CFG.device)
        
        self.fc = nn.Linear(self.hiddendim_fc, n_target)
        self._init_weights(self.fc)

    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        hidden_states = layer_wise_embed.permute(1, 0, 2)
        out = self.attention(hidden_states)
        return self.fc(out)

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

    
class LayerWiseCLSPooling(Pooling):
    def __init__(self, hidden_size, n_layer=4, n_target=CFG.n_targets):
        super(LayerWiseCLSPooling, self).__init__()
        self.n_layer = n_layer
        self.fc = nn.Linear(hidden_size, n_target)
        self._init_weights(self.fc)

    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        embed = layer_wise_embed[-2, :, :]
        return self.fc(embed)
   
    
class ConcatPooling(Pooling):
    def __init__(self, hidden_size, n_layer=4, n_target=CFG.n_targets):
        super(ConcatPooling, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.fc = nn.Linear(hidden_size * n_layer, n_target)
        self._init_weights(self.fc)

    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        embed = layer_wise_embed.permute(1, 0, 2).reshape(-1, self.n_layer * self.hidden_size)

        return self.fc(embed)
    

class LSTMPooling(Pooling):
    def __init__(self, hidden_size, n_layer, n_target=CFG.n_targets, hiddendim_lstm=256, bidirectional=True):
        super(LSTMPooling, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(self.hiddendim_lstm if bidirectional is False else 2 * self.hiddendim_lstm, n_target)
        self._init_weights(self.fc)
    
    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        hidden_states = layer_wise_embed.permute(1, 0, 2)
        out, _ = self.lstm(hidden_states, None)
        out = out[:, -1, :]
        return self.fc(out)


class GRUPooling(Pooling):
    def __init__(self, hidden_size, n_layer, n_target=CFG.n_targets, hiddendim_lstm=256, bidirectional=True):
        super(GRUPooling, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.gru = nn.GRU(self.hidden_size, self.hiddendim_lstm, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(self.hiddendim_lstm if bidirectional is False else 2 * self.hiddendim_lstm, n_target)
        self._init_weights(self.fc)
    
    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        hidden_states = layer_wise_embed.permute(1, 0, 2)
        out, _ = self.gru(hidden_states, None)
        out = out[:, -1, :]
        return self.fc(out)


class DecoupledPooling(nn.Module):
    def __init__(self, pooling_class, hidden_size, n_layer=4):
        super(DecoupledPooling, self).__init__()
        self.n_layer = n_layer
        self.heads = []
        for i in range(CFG.n_targets):
            self.heads.append(pooling_class(hidden_size, n_layer, n_target=1))
        self.heads = nn.ModuleList(self.heads)

    def forward(self, layer_wise_embed):
        # layer_wise_embed: shape -> [n_layer, bs, hidden_size]
        res = []
        for i in range(CFG.n_targets):
            res.append(self.heads[i](layer_wise_embed))
        res = torch.cat(res, 1)

        return res

