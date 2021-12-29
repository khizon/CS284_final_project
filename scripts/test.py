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

from collections import defaultdict

from utils import *
from constants import *
import os
import wandb


if __name__ == '__main__':
    
    user = FILES['USER']
    project = FILES['PROJECT']
    artifact_ver = f'{FILES["MODEL_NAME"]}:{FILES["VERSION"]}'
    model_path = f"{user}/{project}/{artifact_ver}"
    

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
        

    test_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'test.jsonl'),
        tokenizer,
        max_len = CONFIG['MAX_LEN'],
        sample = CONFIG['SAMPLE'],
        title_only = CONFIG['TITLE_ONLY']
    )
    
    # Download Model From Artifacts WandB
    run = wandb.init(project=project, entity=user)
    wandb.watch(model, log='all')
    
    artifact = run.use_artifact(model_path, type='model')
    artifact_dir = artifact.download()
    
    # Load Model From Artifacts
    checkpoint = torch.load(os.path.join('artifacts', artifact_ver, 'torch_checkpoint.bin'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(CONFIG['DEVICE'])

    y_pred, y_test, test_acc = get_predictions(model, test_data_loader)
    
    test_results = {
        'predictions': y_pred,
        'labels': y_test,
        'test_acc' : test_acc
    }
    
    wandb.log({"test acc": test_acc})
    
    if not os.path.exists(os.path.join('results')):
            os.makedirs(os.path.join('results'))

    with open(os.path.join('checkpoint', 'test_results.pickle'), 'wb') as f:
                pickle.dump(test_results, f)

    # Save model to weights and biases
    artifact = wandb.Artifact('test_results', type='results')
    artifact.add_file(os.path.join('checkpoint', 'test_results.pickle'))
    run.log_artifact(artifact)
    run.join()
    run.finish()
    wandb.finish()