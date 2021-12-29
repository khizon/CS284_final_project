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

def train(config = None):
    
    with wandb.init(config=config, entity=FILES['USER']) as run:
        config = wandb.config
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ = torch.manual_seed(config.seed)
        

        # Initialize Model
        tokenizer, model = create_model(config.model_name, config.dropout)
        model.to(device)

        # Initialize Train and Eval data set
        train_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'train.jsonl'),
            tokenizer,
            max_len = config.max_len,
            batch_size = config.batch_size,
            shuffle=True,
            sample = config.sample,
            title_only = config.title_only
        )

        val_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'val.jsonl'),
            tokenizer,
            max_len = config.max_len,
            batch_size = config.batch_size,
            sample = config.sample,
            title_only = config.title_only
        )

        # Optimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr = config.learning_rate)
        total_steps = len(train_data_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps =  int(total_steps * config.warmup),
            num_training_steps = total_steps
        )

        # Save Model Config
        if not os.path.exists(os.path.join('checkpoint')):
                os.makedirs(os.path.join('checkpoint'))
            
        with open(os.path.join('checkpoint', 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=4)

        # Initialize Early Stopping
        best_accuracy = 0.0
        early_stopping = EarlyStopping(patience = config.patience, min_delta = config.min_delta)

        # Training Loop
        wandb.watch(model, log='all')
        for epoch in range(config.epochs):
            print(f'Training {epoch + 1}/{config.epochs}:')

            train_acc, train_loss = train_epoch(
                model, config.model_name, train_data_loader, optimizer, 
                device, scheduler
            )

            val_acc, val_loss = eval_model(model, config.model_name, val_data_loader, device)

            wandb.log({
                "train acc": train_acc,
                "train_loss": train_loss,
                "val acc": val_acc,
                "val_loss": val_loss,
                "epoch" : epoch
            })

            # Checkpoint Best Model
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

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project = FILES['PROJECT'])
    wandb.agent(sweep_id, train, count=5)