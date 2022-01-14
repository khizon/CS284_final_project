from torch.utils.data import Dataset, DataLoader

import transformers

import pandas as pd
import numpy as np
import json

from utils import *
from constants import *
import os
import wandb

def test(config = None):
    with wandb.init(config=config,entity=FILES['USER']) as run:
        config = wandb.config
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(86)

        # Initialize Tokenizer and Model
        tokenizer, model = create_model(config.model_name)
        model.to(device)

        wandb.watch(model, log='all')
        # Initialize test data set
        if config.model_name == 'khizon/bert-unreliable-news-eng-title':
            max_len = 128
            title_only = True
        else:
            max_len = 512
            title_only = False

        test_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'test.jsonl'),
            tokenizer,
            max_len = max_len,
            sample = 10,
            title_only = title_only,
            random_state = config.seed
        )

        y_pred, y_test, test_acc, ave_time = get_predictions(model, config.model_name, test_data_loader, device)

        wandb.log({
            "test acc": test_acc,
            "ave_time": ave_time
        })

        run.join()
        run.finish()

if __name__ == '__main__':
    transformers.logging.set_verbosity_info()
    sweep_config['parameters'] = test_dict
    sweep_id = wandb.sweep(sweep_config, project = FILES['PROJECT'])
    wandb.agent(sweep_id, test)