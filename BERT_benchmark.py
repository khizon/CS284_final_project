from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

from collections import defaultdict

from utils import *
from constants import *

if __name__ == '__main__':

    _ = torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    model = ReliableNewsClassifier(CONFIG['MODEL_NAME'])
    model.to(CONFIG['DEVICE'])

    criterion = nn.BCEWithLogitsLoss().to(CONFIG['DEVICE'])

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

    # Training Loop

    history = defaultdict(list)
    best_accuracy = 0
    early_stopping = EarlyStopping(patience = 3)

    for epoch in range(CONFIG['EPOCHS']):
        print(f'Training: {epoch + 1}/{CONFIG["EPOCHS"]} ------')

        train_acc, train_loss = train_epoch(
            model, train_data_loader, criterion, optimizer, 
            CONFIG['DEVICE'], scheduler
            )

        val_acc, val_loss = eval_model(model, val_data_loader, criterion, CONFIG['DEVICE'])

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
            torch.save(checkpoint, 'best_' + CONFIG['MODEL_NAME'] + '_state.bin')
            best_accuracy = val_acc
        
        torch.save(history, 'train_history.bin')
        #Stop training when accuracy plateus.
        early_stopping(val_acc)
        if early_stopping.early_stop:
            break
