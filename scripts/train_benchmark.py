from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
import json

from collections import defaultdict

from utils import *
from constants import *

if __name__ == '__main__':

    _ = torch.manual_seed(42)

    if CONFIG['MODEL_NAME'] == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        config = BertConfig.from_pretrained(CONFIG['MODEL_NAME'])
        config.num_labels = 1
        model = BertForSequenceClassification(config)
    elif CONFIG['MODEL_NAME'] == 'distilbert-base-cased':
        tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        config = DistilBertConfig.from_pretrained(CONFIG['MODEL_NAME'])
        config.num_labels = 1
        model = DistilBertForSequenceClassification(config)

    model.to(CONFIG['DEVICE'])

    train_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'train.jsonl'),
        tokenizer,
        max_len = CONFIG['MAX_LEN'],
        batch_size = CONFIG['BATCH_SIZE'],
        shuffle=True,
        sample = CONFIG['SAMPLE'],
        title_only = CONFIG['TITLE_ONLY']
    )

    val_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'val.jsonl'),
        tokenizer,
        max_len = CONFIG['MAX_LEN'],
        batch_size = CONFIG['BATCH_SIZE'],
        sample = CONFIG['SAMPLE'],
        title_only = CONFIG['TITLE_ONLY']
    )

    optimizer = AdamW(model.parameters(), lr=CONFIG['LR'])
    total_steps = len(train_data_loader) * CONFIG['EPOCHS']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(total_steps * CONFIG['WARMUP']),
        num_training_steps = total_steps
    )
    
    if not os.path.exists(os.path.join('checkpoint')):
                os.makedirs(os.path.join('checkpoint'))
            
    with open(os.path.join('checkpoint', 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=4)

    # Training Loop

    history = defaultdict(list)
    best_accuracy = 0
    early_stopping = EarlyStopping(patience = CONFIG['PATIENCE'])

    for epoch in range(CONFIG['EPOCHS']):
        print(f'Training: {epoch + 1}/{CONFIG["EPOCHS"]} ------')

        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, 
            CONFIG['DEVICE'], scheduler
            )

        val_acc, val_loss = eval_model(model, val_data_loader, CONFIG['DEVICE'])

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        # Check point best performing model
        if val_acc > best_accuracy:
            checkpoint = {
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'loss' : val_loss,
                'accuracy': val_acc,
                'epoch': epoch
            }
            
            
            torch.save(checkpoint, os.path.join('checkpoint', 'pytorch_checkpoint.pth.tar'))
            best_accuracy = val_acc
        
        if not os.path.exists(os.path.join('results')):
            os.makedirs(os.path.join('results'))
        
        torch.save(history, os.path.join('results','train_history.pth.tar'))
        #Stop training when accuracy plateus.
        early_stopping(val_acc)
        if early_stopping.early_stop:
            break
