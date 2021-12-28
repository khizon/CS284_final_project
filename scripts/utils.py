import os
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

from constants import *

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
    test_df = jsonl_to_df(os.path.join(file_path, 'train.jsonl'))
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
                data_row.content,
                add_special_tokens=True,
                max_length = self.max_token_len,
                return_token_type_ids = True,
                padding = 'max_length',
                truncation = True,
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
Train Function
'''
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    n_examples = 0

    loop = tqdm(data_loader)
    for idx, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
                        labels = labels)

        preds = torch.round(outputs['logits'])
        loss = outputs['loss']

        correct_predictions += (preds == labels).sum().item()
        n_examples += len(labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(train_loss = np.mean(losses), train_acc = float(correct_predictions/n_examples))

    return correct_predictions/n_examples, np.mean(losses)

'''
Distillation Training Function
'''
def sof_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def distill_train_epoch(student_model, teacher_model, train_dataloader, n_gpu):
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, label_ids = batch
        if input_ids.size()[0] != DISTILL_CONFIG['BATCH_SIZE']:
            continue

        att_loss = 0.
        rep_loss = 0.
        cls_loss = 0.

        student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask, is_student=True)

        with torch.no_grad():
            teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

        if not DISTILL_CONFIG['PRED_DISTILL']:
            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)
            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device), student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device), teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                att_loss += tmp_loss

            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
            new_student_reps = student_reps
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss

            loss = rep_loss + att_loss
            tr_att_loss += att_loss.item()
            tr_rep_loss += rep_loss.item()
        else:
            cls_loss = soft_cross_entropy(student_logits / DISTILL_CONFIG['TEMP'], teacher_logits / DISTILL_CONFIG['TEMP'])

            loss = cls_loss
            tr_cls_loss += cls_loss.item()

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += label_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
'''
Evaluation Function
'''
def eval_model(model, data_loader, device):
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

            outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
                        labels = labels)

            preds = torch.round(outputs['logits'])
            loss = outputs['loss']

            correct_predictions += (preds == labels).sum().item()
            n_examples += len(labels)
            losses.append(loss.item())

            loop.set_postfix(val_loss = np.mean(losses), val_acc = float(correct_predictions/n_examples))

    return correct_predictions / n_examples, np.mean(losses)

'''
Get Predictions
'''
def get_predictions(model, data_loader):
    model = model.eval()
    
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, batch in enumerate(loop):

            input_ids = batch["input_ids"].to(CONFIG['DEVICE'])
            attention_mask = batch["attention_mask"].to(CONFIG['DEVICE'])
            token_type_ids = batch['token_type_ids'].to(CONFIG['DEVICE'])
            labels = batch["labels"].to(CONFIG['DEVICE']).unsqueeze(1)

            outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
                        labels = labels)

            preds = torch.round(outputs['logits'])

            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values

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