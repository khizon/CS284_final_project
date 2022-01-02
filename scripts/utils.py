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
def create_model(model_name, dropout, freeze_bert = True):
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
    if freeze_bert:
        for name, param in model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
    return tokenizer, model
'''
Initialize Student Teacher models
'''

def teacher_student_models(model_name):
    pass

    # Download teacher weights from WandB

'''
Train Function
'''
def train_epoch(model, model_name, data_loader, optimizer, device, scheduler, scaler=None):
    model = model.train()
    n_gpu = torch.cuda.device_count()
    losses = []
    correct_predictions = 0
    n_examples = 0

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

        preds = torch.round(outputs['logits'])
        loss = outputs['loss'].mean() if n_gpu > 1 else outputs['loss']

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
Distillation Training Function
'''
def sof_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def distill_train_epoch(student_model, teacher_model, train_dataloader, optimizer, device):
    student_model.train()
    teacher_model.eval()
    n_gpu = torch.cuda.device_count()

    tr_rep_losses = []
    tr_att_losses = []
    correct_predictions = 0
    n_examples = 0

    loop = tqdm(data_loader)
    for step, batch in enumerate(loop):
        att_loss = 0.0
        rep_loss = 0.0

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        student_logits, student_atts, student_reps = student_model(input_ids, token_type_ids, attention_mask, is_student=True)

        with torch.no_grad():
            teacher_logits, teacher_atts, teacher_reps = student_model(input_ids, token_type_ids, attention_mask)


        # Compute Student accuracy
        preds = torch.round(student_logits)
        correct_predictions += (preds == labels).sum().item()
        n_examples += len(labels)

        # Compare student and teacher layers
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

            tmp_loss = MSELoss(student_att, teacher_att)
            att_loss += tmp_loss

        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps

        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            tmp_loss = MSELoss(student_rep, teacher_rep)
            rep_loss += tmp_loss

        loss = rep_loss + att_loss
        tr_rep_losses.append(rep_loss.item())
        tr_att_losses.append(att_loss.item())
        
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(att_loss = np.mean(tr_att_losses), rep_loss = np.mean(tr_rep_losses), train_acc = float(correct_predictions/n_examples))

    return correct_predictions/n_examples, np.mean(tr_att_losses), np.mean(tr_rep_losses)
        
'''
Evaluation Function
'''
def eval_model(model, model_name, data_loader, device):
    model = model.eval()

    losses = []
    correct_predictions = 0
    n_examples = 0

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
            elif model_name == 'tinyBert':
                logits, _, _ = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels)

            # preds = torch.round(outputs['logits'])
            preds = torch.round(logits)
            loss = outputs['loss']

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
            
            timings.append(end-start)

            preds = torch.round(outputs['logits'])
            correct_predictions += (preds == labels).sum().item()
            n_examples += len(labels)

            predictions.extend(preds)
            real_values.extend(labels)

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