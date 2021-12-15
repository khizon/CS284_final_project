import os

CONFIG = {
        'FILE_PATH': os.path.join('data', 'nela_gt_2018_site_split'),
        'MODEL_NAME': 'bert-base-cased',
        # 'DEVICE' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'DEVICE': 'cpu',
        'MAX_LEN': 128,
        'BATCH_SIZE': 8,
        'EPOCHS': 10,
        'LR': 2e-5,
        'WARMUP': 0.1,
        'SAMPLE': 16
    }