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
        'BATCH_SIZE': 8,
        'EPOCHS': 10,
        'LR': 2e-5,
        'WARMUP': 0.1,
        'SAMPLE': 40,
        'TITLE_ONLY': False,
        'DROPOUT': 0.2,
        'PATIENCE': 1
    }

DISTILL_CONFIG = {
        'PRED_DISTILL' : True,
        'FILE_PATH' : os.path.join('data', 'nela_gt_2018_site_split'),
        'DEVICE' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'SEED' : 86,
        'OUTPUT_DIR': '',
        'MAX_LEN' : 512,
        'EPOCHS' : 10,
        'BATCH_SIZE' : 8,
        'SAMPLE' : None,
        'TITLE_ONLY' : False,
        'STUDENT_MODEL' : '',
        'LR' : 2E-5,
        'WARMUP' : 0.1,
        'TEMP' : 1
}