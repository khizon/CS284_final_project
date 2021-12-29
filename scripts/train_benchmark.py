from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
import pickle
import shutil

from collections import defaultdict

from utils import *
from constants import *
import wandb

if __name__ == '__main__':

    _ = torch.manual_seed(42)
    
    wandb.config = CONFIG

    if CONFIG['MODEL_NAME'] == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        config = BertConfig.from_pretrained(CONFIG['MODEL_NAME'])
        config.dropout = CONFIG['DROPOUT']
        config.num_labels = 1
        model = BertForSequenceClassification(config)
    elif CONFIG['MODEL_NAME'] == 'distilbert-base-cased':
        tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        config = DistilBertConfig.from_pretrained(CONFIG['MODEL_NAME'])
        config.num_labels = 1
        config.dropout = CONFIG['DROPOUT']
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
    
    # Save model config
    if not os.path.exists(os.path.join('checkpoint')):
                os.makedirs(os.path.join('checkpoint'))
            
    with open(os.path.join('checkpoint', 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=4)

    # Training Loop
    best_accuracy = 0
    early_stopping = EarlyStopping(patience = CONFIG['PATIENCE'])
    
    # Weights and Biases Set up
    run = wandb.init(project=FILES['PROJECT'], entity=FILES['USER'])
    wandb.watch(model, log='all')
    
    for epoch in range(CONFIG['EPOCHS']):
        print(f'Training: {epoch + 1}/{CONFIG["EPOCHS"]} ------')

        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, 
            CONFIG['DEVICE'], scheduler
            )

        val_acc, val_loss = eval_model(model, val_data_loader, CONFIG['DEVICE'])
        
        wandb.log({"train loss": train_loss,
                   "val loss": val_loss,
                   "train acc": train_acc,
                   "val acc": val_acc
                  })

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
            
            # model.save_pretrained(os.path.join('checkpoint'))
            torch.save(checkpoint, os.path.join('checkpoint', 'torch_checkpoint.bin'))
            best_accuracy = val_acc
                
        #Stop training when accuracy plateus.
        early_stopping(val_acc)
        if early_stopping.early_stop:
            break
    
    
    # Save model to weights and biases
    artifact = wandb.Artifact(FILES['MODEL_NAME'], type='model')
    artifact.add_file(os.path.join('checkpoint', 'torch_checkpoint.bin'))
    artifact.add_file(os.path.join('checkpoint', 'config.json'))

    run.log_artifact(artifact)
    run.join()
    run.finish()
    wandb.finish()
