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
# from sklearn.metrics import classification_report, confusion_matrix

from collections import defaultdict

from utils import *
from constants import *
import os


if __name__ == '__main__':

    if CONFIG['MODEL_NAME'] == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        model = BertForSequenceClassification.from_pretrained(os.path.join('checkpoint'))
    elif CONFIG['MODEL_NAME'] == 'distilbert-base-cased':
        tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
        model = DistilBertForSequenceClassification.from_pretrained(os.path.join('checkpoint'))
        

    test_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'test.jsonl'),
        tokenizer,
        max_len = CONFIG['MAX_LEN'],
        sample = CONFIG['SAMPLE'],
        title_only = CONFIG['TITLE_ONLY']
    )
    
    model.to(CONFIG['DEVICE'])

    y_pred, y_test = get_predictions(model, test_data_loader)

    # print(classification_report(y_test, y_pred))
    test_results = {
        'predictions': y_pred,
        'labels': y_test
    }
    
    if not os.path.exists(os.path.join('results')):
            os.makedirs(os.path.join('results'))

    # torch.save(test_results, os.path.join('results', 'test_results.bin'))
    with open(os.path.join('results', 'test_results.pickle'), 'wb') as f:
                pickle.dump(test_results, f)