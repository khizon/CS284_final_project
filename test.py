from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from collections import defaultdict

from utils import *
from constants import *


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    model = ReliableNewsClassifier(CONFIG['MODEL_NAME'])
    model.to(CONFIG['DEVICE'])

    test_data_loader = create_reliable_news_dataloader(
        os.path.join(CONFIG['FILE_PATH'], 'test.jsonl'),
        tokenizer,
        sample = CONFIG['SAMPLE']
    )

    checkpoint = torch.load('best_' + CONFIG['MODEL_NAME'] + '_state.bin', map_location=torch.device(CONFIG['DEVICE']))
    model.load_state_dict(checkpoint['state_dict'])

    y_titles, y_pred, y_test = get_predictions(model, test_data_loader)

    print(classification_report(y_test, y_pred))