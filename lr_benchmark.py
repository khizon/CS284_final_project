from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from collections import defaultdict

from utils import *

if __name__ == '__main__':

    _ = torch.manual_seed(42)

    CONFIG = {
        'FILE_PATH': os.path.join('data', 'nela_gt_2018_site_split'),
        'MODEL_NAME': 'bert-base-cased',
        'DEVICE' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # 'DEVICE': 'cpu',
        'MAX_LEN': 128,
        'BATCH_SIZE': 8,
        'EPOCHS': 10,
        'LR': 2e-5
    }

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    model = ReliableNewsClassifier(CONFIG['MODEL_NAME'])
    model.to(CONFIG['DEVICE'])

    criterion = nn.BCEWithLogitsLoss().to(CONFIG['DEVICE'])

    train_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'train.jsonl'),
        tokenizer,
        shuffle=True,
        sample=30
    )

    val_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'val.jsonl'),
        tokenizer,
        sample=30
    )

    optimizer =AdamW(model.parameters(), lr=CONFIG['LR'])
    total_steps = len(train_data_loader) * CONFIG['EPOCHS']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = total_steps//10,
        num_training_steps = total_steps
    )

    sample = next(iter(train_data_loader))
    
    # print(sample['labels'])
    # output = model(input_ids = sample['input_ids'], attention_mask = sample['attention_mask'])
    # print(f'output: {output}')

    train_acc, train_loss = train_epoch(
        model, train_data_loader, criterion, optimizer, 
        CONFIG['DEVICE'], scheduler
        )
    print(f'acc: {train_acc} loss: {train_loss}')

    val_acc, val_loss = eval_model(model, val_data_loader, criterion, CONFIG['DEVICE'])
    print(f'acc: {val_acc} loss: {val_loss}')