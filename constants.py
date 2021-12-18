import os
import torch
'''
Config settings for the experiments
To use the whole dataset: SAMPLE = None
To use the concatenated title + content: TITLE_ONLY = False
'''
CONFIG = {
        'FILE_PATH': os.path.join('data', 'nela_gt_2018_site_split'),
        'MODEL_NAME': 'bert-base-cased',
        'DEVICE' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # 'DEVICE': 'cpu',
        'MAX_LEN': 512,
        'BATCH_SIZE': 32,
        'EPOCHS': 10,
        'LR': 5e-5,
        'WARMUP': 0.1,
        'SAMPLE': None,
        'TITLE_ONLY': True
    }