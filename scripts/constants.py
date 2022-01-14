import os
import torch
import math

'''
Config settings for the experiments
To use the whole dataset: SAMPLE = None
To use the concatenated title + content: TITLE_ONLY = False
Dropout parameter only limited to classifier dropout.


Files settings for saving checkpoints to WandB
Change MODEL_NAME when using different settings
'''

FILES = {
        'PROJECT' : 'UnreliableNews',
        'MODEL_NAME' : 'BERT-title-content-benchmark-test',
        'VERSION' : 'v6',
        'USER' : 'khizon',
    }

# Random Search Optimization
sweep_config = {
    'name' : 'cpu_testing',
    'method' : 'grid'
    }


# Hyperparameters kept constant
parameter_dict = {
    'learning_rate' : {'value' : 2e-5}, # modify
    'epochs' : {'value' : 10},
    'warmup' : {'value' : 0.06},
    'max_len' : {'value' : 512}, # modify
    'batch_size' : {'value' : 8}, # modify
    'dropout' : {'value' : 0.10},
    'weight_decay' : {'value' : 0.10},
    'patience' : {'value': 10},
    'min_delta' : {'value' : 0.005}, 
    'sample' : {'value' : 32}, # set to false for real training
    'title_only' : {'value' : False}, # modify
    'freeze_bert' : {'value' : False},
    'dataset_path' : {'value' : os.path.join('data', 'nela_gt_2018_site_split')},
    'model_name' : {'value' : 'bert-base-cased'}, # modify
    'seed' : {'value' : 86}
}

distill_dict = {
    'learning_rate' : {'value' : 2e-5}, # modify
    'epochs' : {'value' : 10},
    'warmup' : {'value' : 0.06},
    'weight_decay' : {'value' : 0.10},
    'patience' : {'value': 3},
    'min_delta' : {'value' : 0.005},
    'batch_size' : {'value' : 8}, # modify
    'sample' : {'value' : 8}, # set to false for real training
    'dataset_path' : {'value' : os.path.join('data', 'nela_gt_2018_site_split')},
    'student_model' : {'value' : 'distilbert-base-cased'},
    'num_layers' : {'value' : 4},
    'teacher_model' : {'value' : 'BERT-title-content-benchmark:v0'},
    'seed' : {'value' : 86},
    'pred_distill' : {'value' : True},
    'alpha' : {'value' : 0.5},
    'do_eval' : {'value' : True}
}

test_dict = {
    'model_name' : {'values' : ['khizon/bert-unreliable-news-eng', 'khizon/distilbert-unreliable-news-eng-4L', 'khizon/distilbert-unreliable-news-eng-6L']},
    'seed' : {'values' : [86, 42, 0, 1, 99]},
    'dataset_path' : {'value' : os.path.join('data', 'nela_gt_2018_site_split')},
    'batch_size' : {'value' : 8},
}
