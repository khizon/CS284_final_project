import os
import pandas as pd
import numpy as np
import random
import json
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from transformer import TinyBertForSequenceClassification

import pandas as pd
import numpy as np
import wandb
import time

# from constants import *
'''
RNG seed
'''

def seed_everything(seed=86):
    _ = torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

'''
Convert jsonl files to pandas dataset
'''
def jsonl_to_df(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()

    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']

    df_inter['json_element'].apply(json.loads)

    return pd.json_normalize(df_inter['json_element'].apply(json.loads))

'''
Load all datasets into one.
Use this for visualization and EDA
'''
def load_dataset(file_path):
    train_df = jsonl_to_df(os.path.join(file_path, 'train.jsonl'))
    train_df['split'] = 'train'
    val_df = jsonl_to_df(os.path.join(file_path, 'val.jsonl'))
    val_df['split'] = 'val'
    test_df = jsonl_to_df(os.path.join(file_path, 'test.jsonl'))
    test_df['split'] = 'test'

    df = pd.concat([train_df, val_df, test_df])
    pd.concat([train_df, val_df, test_df])
    df.fillna('', inplace=True)
    print(df.sample(5))
    return df

'''
DataSet class
'''
class ReliableNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len = 128, title_only=True):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.title_only = title_only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        
        labels = data_row.label
        if self.title_only:
            encoding = self.tokenizer.encode_plus(
                data_row.title,
                add_special_tokens=True,
                max_length = self.max_token_len,
                return_token_type_ids = True,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                return_tensors = 'pt'
            )
        else:
            encoding = self.tokenizer.encode_plus(
                data_row.title,
                ' [SEP] ' + data_row.content,
                add_special_tokens=True,
                max_length = self.max_token_len,
                return_token_type_ids = True,
                padding = 'max_length',
                truncation = 'only_second',
                return_attention_mask = True,
                return_tensors = 'pt'
            )

        return dict(
            # text = data_row.title + ' ' + data_row.content,
            input_ids = encoding['input_ids'].flatten(),
            attention_mask = encoding['attention_mask'].flatten(),
            token_type_ids = encoding['token_type_ids'].flatten(),
            labels = torch.tensor(labels, dtype=torch.float32)
        )

def create_reliable_news_dataloader(file_path, tokenizer, max_len=128, batch_size=8, shuffle=False, sample = None, title_only = True):

    print(f'Max token length: {max_len} Batch size: {batch_size} Shuffle: {shuffle} Title only: {title_only}')
    df = jsonl_to_df(file_path)
    
    # Load only a partial dataset
    if sample:
        df = df.sample(sample)
    
    ds = ReliableNewsDataset(df, tokenizer, max_token_len = max_len, title_only = title_only)
    return DataLoader(ds, batch_size = batch_size, shuffle=shuffle)

'''
Create Model
'''
def create_model(model_name, dropout=0.1, freeze_bert = False, distill = False, student_dim = None):
    if model_name == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name)
        config.dropout = dropout
        config.num_labels = 1
        model = BertForSequenceClassification(config)
    elif model_name == 'distilbert-base-cased':
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        config = DistilBertConfig.from_pretrained(model_name)
        config.dropout = dropout
        config.num_labels = 1
        model = DistilBertForSequenceClassification(config)
    elif model_name == 'tiny-bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model_path = os.path.join('artifacts', '2nd_General_TinyBERT_6L_768D')
        model = TinyBertForSequenceClassification.from_pretrained(model_path, num_labels = 1)
    if freeze_bert:
        for name, param in model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
    if distill:
        model.config.output_attentions = True
        model.config.output_hidden_states = True
    return tokenizer, model

'''
Train Function
'''
def train_epoch(model, model_name, data_loader, optimizer, device, scheduler, scaler=None):
    model = model.train()
    n_gpu = torch.cuda.device_count()
    losses = []
    correct_predictions = 0
    n_examples = 0
    
    if model_name == 'tiny-bert':
        criterion = torch.nn.BCEWithLogitsLoss()

    loop = tqdm(data_loader)
    for idx, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        model.zero_grad()
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        if model_name == 'bert-base-cased':
            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            labels = labels)
        elif model_name == 'distilbert-base-cased':
            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels)
        elif model_name == 'tiny-bert':
            logits, _, _ = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            labels = labels)

        if model_name == 'tiny-bert':
            loss = criterion(logits, labels)
            preds = torch.round(logits)
        else:
            loss = outputs['loss']
            preds = torch.round(outputs['logits'])

        if n_gpu > 1:
            loss = loss.mean()
        

        correct_predictions += (preds == labels).sum().item()
        n_examples += len(labels)
        losses.append(loss.item())

        # scaler.scale(loss).backward()
        loss.backward()
        # Unscales the gradients of optimizer's assigned params in-place
        # scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        scheduler.step()

        loop.set_postfix(train_loss = np.mean(losses), train_acc = float(correct_predictions/n_examples))

    return correct_predictions/n_examples, np.mean(losses)

'''
Soft Cross Entropy
'''
def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()
'''
Distillation Training Function
'''
def distill_train_epoch(student_model, teacher_model, data_loader, optimizer, device, alpha=0.5, pred_distill=False):
    student_model.train()
    teacher_model.eval()
    n_gpu = torch.cuda.device_count()

    losses = []
    correct_predictions = 0
    n_examples = 0

    loss_mse = MSELoss()

    loop = tqdm(data_loader)
    for step, batch in enumerate(loop):
        att_loss = 0.0
        rep_loss = 0.0
        cls_loss = 0.0
        stud_loss = 0.0

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        student_logits, student_atts, student_reps = student_model(input_ids, token_type_ids, attention_mask, is_student=True)

        with torch.no_grad():
            outputs = teacher_model(input_ids, token_type_ids, attention_mask)
            teacher_logits = outputs['logits']
            teacher_atts = outputs['attentions']
            teacher_reps = outputs['hidden_states']
        
        # print(f'Student Logits: {student_logits} Teacher Logits: {teacher_logits} Labels: {labels}')
        # break

        if pred_distill:
            # distill_loss = loss_mse(student_logits, teacher_logits)
            # student_loss = loss_mse(student_logits, labels)
            for student_logit, label in zip(student_logits, labels):
                tmp_loss = loss_mse(student_logit, label)
                stud_loss += tmp_loss
            for student_logit, teacher_logit in zip(student_logits, teacher_logits):
                tmp_loss = loss_mse(student_logit, teacher_logit)
                cls_loss += tmp_loss
            loss = alpha * cls_loss + (1-alpha) * stud_loss
        else:
            # Compare student and teacher attention layers
            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)
            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                            student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                            teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                att_loss += tmp_loss

            # Compare teacher and student hidden representation
            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
            new_student_reps = student_reps

            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss

            loss = rep_loss + att_loss
        # tr_rep_losses.append(rep_loss.item())
        # tr_att_losses.append(att_loss.item())
        
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        losses.append(loss.item())
        
        loss.backward()
        
        # Compute Student accuracy
        preds = torch.round(student_logits)
        correct_predictions += (preds == labels).sum().item()
        n_examples += len(labels)
        
        optimizer.step()
        optimizer.zero_grad()
        
        loop.set_postfix(train_loss = np.mean(losses), train_acc = float(correct_predictions/n_examples))

    return correct_predictions/n_examples, np.mean(losses)
        
'''
Evaluation Function
'''
def eval_model(model, model_name, data_loader, device):
    model = model.eval()
    n_gpu = torch.cuda.device_count()

    losses = []
    correct_predictions = 0
    n_examples = 0
    
    if model_name == 'tiny-bert':
        criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            if model_name == 'bert-base-cased':
                outputs = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                labels = labels)
            elif model_name == 'distilbert-base-cased':
                outputs = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels)
            elif model_name == 'tiny-bert':
                logits, _, _ = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                labels = labels)

            if model_name == 'tiny-bert':
                loss = criterion(logits, labels)
                preds = torch.round(logits)
            else:
                loss = outputs['loss']
                preds = torch.round(outputs['logits'])

            if n_gpu > 1:
                loss = loss.mean()

            correct_predictions += (preds == labels).sum().item()
            n_examples += len(labels)
            losses.append(loss.item())

            loop.set_postfix(val_loss = np.mean(losses), val_acc = float(correct_predictions/n_examples))

    return correct_predictions / n_examples, np.mean(losses)

'''
Get Predictions
'''
def get_predictions(model, model_name, data_loader, device):
    model = model.eval()
    
    predictions = []
    real_values = []
    correct_predictions = 0
    n_examples = 0
    
    # Timer
    timings = []
    
    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, batch in enumerate(loop):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            if model_name == 'bert-base-cased':
                start = time.perf_counter()
                outputs = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                labels = labels)
                end = time.perf_counter()
            elif model_name == 'distilbert-base-cased':
                start = time.perf_counter()
                outputs = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels)
                end = time.perf_counter()
            elif model_name == 'tiny-bert':
                start = time.perf_counter()
                logits, _, _ = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                labels = labels)
                end = time.perf_counter()
            
            timings.append(end-start)

            if model_name == 'tiny-bert':
                preds = torch.round(logits)
            else:
                preds = torch.round(outputs['logits'])

            correct_predictions += (preds == labels).sum().item()
            n_examples += len(labels)

            predictions.extend(preds)
            real_values.extend(labels)
            loop.set_postfix(test_acc = float(correct_predictions/n_examples))

    # print(f'correct: {correct_predictions} n: {n_examples}')
    predictions = torch.stack(predictions).cpu().detach().tolist()
    real_values = torch.stack(real_values).cpu().detach().tolist()
    return predictions, real_values, correct_predictions / n_examples, (np.mean(timings))

'''
Early Stopping
'''
class EarlyStopping():
    """
    Early stopping to stop the training when the accuracy does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.05):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif val_acc - self.best_acc > self.min_delta:
            self.best_acc = val_acc
            # reset counter if validation loss improves
            self.counter = 0
        elif val_acc - self.best_acc < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True