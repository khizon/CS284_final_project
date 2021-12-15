import os
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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
    print(df.sample(5))
    return df

'''
DataSet class
'''
class ReliableNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        title = data_row.title
        labels = data_row.label

        encoding = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length = self.max_token_len,
            return_token_type_ids = False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return dict(
            title = title,
            input_ids = encoding['input_ids'].flatten(),
            attention_mask = encoding['attention_mask'].flatten(),
            labels = torch.tensor(labels, dtype=torch.float32)
        )

def create_reliable_news_dataloader(file_path, tokenizer, max_len=128, batch_size=8, shuffle=False, sample = None):
    df = jsonl_to_df(file_path)
    
    # Load only a partial dataset
    if sample:
        df = df.sample(sample)
    
    ds = ReliableNewsDataset(df, tokenizer, max_token_len = max_len)
    return DataLoader(ds, batch_size = batch_size, shuffle=shuffle)

'''
Model Class
'''
class ReliableNewsClassifier(nn.Module):
    def __init__(self, model_name):
        super(ReliableNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(x.pooler_output)
        return self.classifier(x)

'''
Train Function
'''
def train_epoch(model, data_loader, criterion, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    n_examples = 0

    loop = tqdm(data_loader)
    for idx, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        outputs = model(input_ids = input_ids, attention_mask = attention_mask)

        preds = torch.round(torch.sigmoid(outputs))
        loss = criterion(outputs, labels)

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
Evaluation Function
'''
def eval_model(model, data_loader, criterion, device):
    model = model.eval()

    losses = []
    correct_predictions = 0
    n_examples = 0

    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.round(torch.sigmoid(outputs))

            loss = criterion(outputs, labels)

            # correct_predictions += torch.sum(preds == labels.unsqueeze(1))
            correct_predictions += (preds == labels).sum().item()
            n_examples += len(labels)
            losses.append(loss.item())

            loop.set_postfix(val_loss = np.mean(losses), val_acc = float(correct_predictions/n_examples))

    return correct_predictions / n_examples, np.mean(losses)

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