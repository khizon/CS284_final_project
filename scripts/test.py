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

def test(config = None):
    user = FILES['USER']
    project = FILES['PROJECT']
    artifact_ver = f'{FILES["MODEL_NAME"]}:{FILES["VERSION"]}'
    model_path = f"{user}/{project}/{artifact_ver}"

    with wandb.init(config=config, project=FILES['PROJECT'], entity=FILES['USER']):
        _ = torch.manual_seed(config.seed)

        # Initialize Tokenizer and Model
        tokenizer, model = create_model(config.model_name, config.droput)

        # Initialize test data set
        test_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'test.jsonl'),
            tokenizer,
            max_len = config.max_len,
            sample = config.sample,
            title_only = config.title_only
        )

        wandb.watch(model, log='all')
        # Load Model From Artifacts
        artifact = run.use_artifact(model_path, type='model')
        artifact_dir = artifact.download()
        checkpoint = torch.load(os.path.join('artifacts', artifact_ver, 'torch_checkpoint.bin'))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config.device)

        y_pred, y_test, test_acc, ave_time = get_predictions(model, config.model_name, test_data_loader, config.device)

        test_results = {
            'predictions': y_pred,
            'labels': y_test,
            'test_acc' : test_acc,
            'ave_time' : ave_time
        }

        wandb.log({
            "test acc": test_acc,
            "ave_time": ave_time
        })

        if not os.path.exists(os.path.join('checkpoint')):
            os.makedirs(os.path.join('checkpoint'))

        with open(os.path.join('checkpoint', 'test_results.pickle'), 'wb') as f:
                    pickle.dump(test_results, f)

        # Save model to weights and biases
        artifact = wandb.Artifact('test_results', type='results')
        artifact.add_file(os.path.join('checkpoint', 'test_results.pickle'))
        run.log_artifact(artifact)
        run.join()
        run.finish()

if __name__ == '__main__':
    wandb.agent(sweep_id, test, count=1)