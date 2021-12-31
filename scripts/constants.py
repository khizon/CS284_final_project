import os
import torch
import math

'''
Config settings for the experiments
To use the whole dataset: SAMPLE = None
To use the concatenated title + content: TITLE_ONLY = False

Files settings for saving checkpoints to WandB
Change MODEL_NAME when using different settings
'''

FILES = {
        'PROJECT' : 'UnreliableNews',
        'MODEL_NAME' : 'BERT-title-content-benchmark',
        'VERSION' : 'v6',
        'USER' : 'khizon',
    }

DISTILL_CONFIG = {
        'PRED_DISTILL' : True,
        'FILE_PATH' : os.path.join('data', 'nela_gt_2018_site_split'),
        'DEVICE' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'SEED' : 86,
        'OUTPUT_DIR': '',
        'MAX_LEN' : 512,
        'EPOCHS' : 5,
        'BATCH_SIZE' : 8,
        'SAMPLE' : None,
        'TITLE_ONLY' : False,
        'STUDENT_MODEL' : '',
        'LR' : 2E-5,
        'WARMUP' : 0.1,
        'TEMP' : 1
}

# Random Search Optimization
sweep_config = {'method' : 'random'}


# Hyperparameters kept constant
parameter_dict = {
    'learning_rate' : {'value' : 2e-5}, # modify
    'epochs' : {'value' : 10},
    'warmup' : {'value' : 0.06},
    'max_len' : {'value' : 512}, # modify
    'batch_size' : {'value' : 8}, # modify
    'dropout' : {'value' : 0.10},
    'weight_decay' : {'value' : 0.10},
    'patience' : {'value': 3},
    'min_delta' : {'value' : 0.005}, 
    'sample' : {'value' : False}, # set to false for real training
    'title_only' : {'value' : False}, # modify
    'freeze_bert' : {'value' : False},
    'dataset_path' : {'value' : os.path.join('data', 'nela_gt_2018_site_split')},
    'model_name' : {'value' : 'bert-base-cased'}, # modify
    'seed' : {'value' : 86}
}

sweep_config['parameters'] = parameter_dict
