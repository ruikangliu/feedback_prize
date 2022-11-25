from config import CFG, config_setup
from utils import MCRMSE, get_score, get_logger, collate, AverageMeter, asMinutes, timeSince, seed_everything, RMSELoss
from adv_training import FGM, AWP
from model import FB3Model, re_initializing_layer
from dataset import FB3TrainDataset

import os
import argparse
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

import scipy as sp
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


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    losses = AverageMeter()
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = CFG.apex)
    start = end = time.time()
    global_step = 0
    if CFG.fgm:
        fgm = FGM(model)
    if CFG.awp:
        awp = AWP(model,
                  optimizer, 
                  adv_lr = CFG.adv_lr, 
                  adv_eps = CFG.adv_eps, 
                  scaler = scaler)
    for step, (inputs, labels) in enumerate(train_loader):
        # attention_mask = inputs['attention_mask'].to(device)
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled = CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        if CFG.unscale:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        #Fast Gradient Method (FGM)
        if CFG.fgm:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled = CFG.apex):
                y_preds = model(inputs)
                loss_adv = criterion(y_preds, labels)
                loss_adv.backward()
            fgm.restore()
            
        #Adversarial Weight Perturbation (AWP)
        if CFG.awp:
            # loss_awp = awp.attack_backward(inputs, labels, attention_mask, step + 1)
            # loss_awp.backward()
            # awp._restore()
            pass
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f} '
                  'LR: {lr:.8f} '
                  .format(epoch + 1, step, len(train_loader), remain = timeSince(start, float(step + 1)/len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]
                         )
                 )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))
                         )
                 )
    return losses.avg, np.concatenate(preds)


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    
    train_folds = folds[folds['fold'] != fold].reset_index(drop = True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop = True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = FB3TrainDataset(CFG, train_folds)
    valid_dataset = FB3TrainDataset(CFG, valid_folds)
    
    train_loader = DataLoader(train_dataset,
                              batch_size = CFG.batch_size,
                              shuffle = True, 
                              num_workers = CFG.num_workers,
                              pin_memory = True, 
                              drop_last = True
                             )
    valid_loader = DataLoader(valid_dataset,
                              batch_size = CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers,
                              pin_memory=True, 
                              drop_last=False)

    model = FB3Model(CFG, config_path = None, pretrained = True)
    if CFG.reinit:
        model = re_initializing_layer(model, model.config, CFG.reinit_n)
        
    #os.makedirs(CFG.OUTPUT_DIR + 'config/', exist_ok = True)
    #torch.save(model.config, CFG.OUTPUT_DIR + 'config/config.pth')
    model.to(CFG.device)
    
    def get_optimizer_params(model,
                             encoder_lr,
                             decoder_lr,
                             weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr,
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr,
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr,
             'weight_decay': 0.0}
        ]
        return optimizer_parameters
    
    #llrd
    def get_optimizer_grouped_parameters(model, 
                                         layerwise_lr,
                                         layerwise_weight_decay,
                                         layerwise_lr_decay):
        
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                         "weight_decay": 0.0,
                                         "lr": layerwise_lr,
                                        },]
        # initialize lrs for every layer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = layerwise_lr
        for layer in layers:
            optimizer_grouped_parameters += [{"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                                              "weight_decay": layerwise_weight_decay,
                                              "lr": lr,
                                             },
                                             {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                                              "weight_decay": 0.0,
                                              "lr": lr,
                                             },]
            lr *= layerwise_lr_decay
        return optimizer_grouped_parameters
    
    if CFG.llrd:
        from transformers import AdamW
        grouped_optimizer_params = get_optimizer_grouped_parameters(model, 
                                                                    CFG.layerwise_lr, 
                                                                    CFG.layerwise_weight_decay, 
                                                                    CFG.layerwise_lr_decay)
        optimizer = AdamW(grouped_optimizer_params,
                          lr = CFG.layerwise_lr,
                          eps = CFG.layerwise_adam_epsilon,
                          correct_bias = not CFG.layerwise_use_bertadam)
    else:
        from torch.optim import AdamW
        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=CFG.encoder_lr, 
                                                    decoder_lr=CFG.decoder_lr,
                                                    weight_decay=CFG.weight_decay)
        optimizer = AdamW(optimizer_parameters, 
                          lr=CFG.encoder_lr,
                          eps=CFG.eps,
                          betas=CFG.betas)
    
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = cfg.num_warmup_steps, 
                num_training_steps = num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = cfg.num_warmup_steps, 
                num_training_steps = num_train_steps,
                num_cycles = cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    
    if CFG.loss_func == 'SmoothL1':
        criterion = nn.SmoothL1Loss(reduction='mean')
    elif CFG.loss_func == 'RMSE':
        criterion = RMSELoss(reduction='mean')
    
    best_score = np.inf
    best_train_loss = np.inf
    best_val_loss = np.inf
    
    epoch_list = []
    epoch_avg_loss_list = []
    epoch_avg_val_loss_list = []
    epoch_score_list = []
    epoch_scores_list = []

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, CFG.device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        
        epoch_list.append(epoch+1)
        epoch_avg_loss_list.append(avg_loss)
        epoch_avg_val_loss_list.append(avg_val_loss)
        epoch_score_list.append(score)
        epoch_scores_list.append(scores)
        
        if best_score > score:
            best_score = score
            best_train_loss = avg_loss
            best_val_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        CFG.OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            
        if CFG.save_all_models:
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        CFG.OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_epoch{epoch + 1}.pth")

    predictions = torch.load(CFG.OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location = torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    
    df_epoch = pd.DataFrame({'epoch' : epoch_list,
                             'MCRMSE' : epoch_score_list,
                             'train_loss' : epoch_avg_loss_list, 
                             'val_loss' : epoch_avg_val_loss_list})
    df_scores = pd.DataFrame(epoch_scores_list)
    df_scores.columns = CFG.target_cols
    
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_train_loss, best_val_loss, valid_folds, pd.concat([df_epoch, df_scores], axis = 1)


def get_result(oof_df, fold, best_train_loss, best_val_loss):
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
    _output_log = pd.DataFrame([CFG.identifier, CFG.model, CFG.cv_seed, CFG.seed, fold, 'best', score, best_train_loss, best_val_loss] + scores).T
    _output_log.columns = ['file', 'model', 'cv_seed', 'seed', 'fold', 'epoch', 'MCRMSE', 'train_loss', 'val_loss'] + CFG.target_cols
    return _output_log


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", help="microsoft/deberta-v3-base", default=None, type=str)
    parser.add_argument("--pooling", help="mean", default=None, type=str)
    parser.add_argument("--loss_func", help="SmoothL1 / RMSE", default=None, type=str)
    parser.add_argument("--decouple_pooling", help="decoupled pooling for different targets", action='store_true')
    parser.add_argument("--gpu_id", default=None, type=str)
    parser.add_argument("--note", default='', type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--zip_res", action='store_true')

    args = parser.parse_args()
    return args

def update_config(cfg, args):
    args = vars(args)
    for k, v in args.items():
        if v is not None:
            exec(f'cfg.{k} = args["{k}"]')


if __name__ == '__main__':
    args = parse_args()
    update_config(CFG, args)
    
    config_setup()
    
    if args.zip_res is True:
        pth_path = os.path.join(CFG.OUTPUT_DIR, '*.pth')
        id = CFG.identifier.replace('/', '_')
        log_path = os.path.join(CFG.OUTPUT_DIR, 'train.log')
        os.system(f'tar -zcv -f {id}.tar.gz {pth_path} {log_path}')
        print(f'{id}.tar.gz is Ready!')
    else:
        # Logger
        LOGGER = get_logger(CFG.log_filename)
        LOGGER.info(f'OUTPUT_DIR: {CFG.OUTPUT_DIR}')
        CFG.LOGGER = LOGGER
        
        # Tokenizer
        CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model)
        CFG.tokenizer.save_pretrained(CFG.OUTPUT_DIR + 'tokenizer')
        # max_len
        if 'deberta' in CFG.model:  # and 'large' not in CFG.model:
            lengths = []
            tk0 = tqdm(CFG.df_train['full_text'].fillna('').values, total=len(CFG.df_train))
            for text in tk0:
                length = len(CFG.tokenizer(text, add_special_tokens=False)['input_ids'])
                lengths.append(length)
            CFG.max_len = max(lengths) + 2
        LOGGER.info(f'max_len: {CFG.max_len}')

        if CFG.train:
            output_log = pd.DataFrame()
            oof_df = pd.DataFrame()
            train_loss_list = []
            val_loss_list = []
            for fold in range(CFG.n_fold):
                if fold in CFG.trn_fold:
                    best_train_loss, best_val_loss, _oof_df, df_epoch_scores = train_loop(CFG.df_train, fold)
                    train_loss_list.append(best_train_loss)
                    val_loss_list.append(best_val_loss)
                    oof_df = pd.concat([oof_df, _oof_df])
                    LOGGER.info(f"========== fold: {fold} result ==========")

                    df_epoch_scores['file'] = CFG.identifier
                    df_epoch_scores['model'] = CFG.model
                    df_epoch_scores['cv_seed'] = CFG.cv_seed
                    df_epoch_scores['seed'] = CFG.seed
                    df_epoch_scores['fold'] = fold
                    df_epoch_scores = df_epoch_scores[['file', 'model', 'cv_seed', 'seed', 'fold', 'epoch', 'MCRMSE', 'train_loss', 'val_loss'] + CFG.target_cols]

                    _output_log = get_result(_oof_df, fold, best_train_loss, best_val_loss)
                    output_log = pd.concat([output_log, df_epoch_scores, _output_log])

            oof_df = oof_df.reset_index(drop=True)
            LOGGER.info(f"========== CV ==========")
            _output_log = get_result(oof_df, 'OOF', np.mean(train_loss_list), np.mean(val_loss_list))
            output_log = pd.concat([output_log, _output_log])
            output_log.to_csv(os.path.join(CFG.OUTPUT_DIR, f'output.csv'), index=False)
            oof_df.to_pickle(os.path.join(CFG.OUTPUT_DIR, 'oof_df.pkl'), protocol = 4)