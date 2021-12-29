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
        'PROJECT' : 'BERT-benchmark',
        'MODEL_NAME' : 'BERT-title-only',
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

# Hyperparameters with discrete (uniform sampling)
parameter_dict = {
    'dropout' : {
        'values' : [0.1, 0.2, 0.3]
    },
    'batch_size': {
        'values' : [8, 16, 32]
    }
}

sweep_config['parameters'] = parameter_dict

# Hyperparameters kept constant
parameter_dict.update({
    'epochs' : {'value' : 10},
    'warmup' : {'value' : 0.1},
    'max_len' : {'value' : 128},
    'patience' : {'value': 3},
    'min_delta' : {'value' : 0.005}, # 0.5%
    'sample' : {'value' : False},
    'title_only' : {'value' : True},
    'dataset_path' : {'value' : os.path.join('data', 'nela_gt_2018_site_split')},
    'model_name' : {'value' : 'bert-base-cased'},
    'seed' : {'value' : 86}
})

# Hyperparameters with distribution
parameter_dict.update({
    'learning_rate' : {
        'distribution' : 'uniform',
        'min' : 0,
        'max' : 5e-5
    }
})
