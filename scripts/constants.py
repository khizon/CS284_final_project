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
        'PROJECT' : 'BERT-test',
        'MODEL_NAME' : 'BERT-title-only',
        'VERSION' : 'v2',
        'USER' : 'khizon',
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

# Random Search Optimization
sweep_config = {'method' : 'random'}

# Hyperparameters with discrete (uniform sampling)
parameters_dict = {
    'dropout' : {
        'values' : [0.1, 0.2, 0.3]
    }
}

sweep_config['parameters'] = parameters_dict

# Hyperparameters kept constant
parameters_dict.update({
    'epochs' : {'value' : 10},
    'warmup' : {'value' : 0.1},
    'max_len' : {'value' : 128},
    'patience' : {'value': 3},
    'min_delta' : {'value' : 0.005}, # 0.5%
    'sample' : {'value' : True},
    'title_only' : {'value' : True},
    'dataset_path' : {'value' : os.path.join('data', 'nela_gt_2018_site_split')},
    'model_name' : {'value' : 'bert-base-cased'},
    'device' : {'value' : torch.device("cuda" if torch.cuda.is_available() else "cpu")},
    'seed' : {'value' : 86}
})

# Hyperparameters with distribution
parameter_dict.update({
    'learning_rate' : {
        'distribution' : 'uniform',
        'min' : 2e-5,
        'max' : 10e-5
    },

    'batch_size' : {
        'distribution' : 'q_log_uniform',
        'q' : 1,
        'min' : math.log(8),
        'max' : math.log(32)
    }
})

sweep_id = wandb.sweep(sweep_config, project = FILES['PROJECT'])