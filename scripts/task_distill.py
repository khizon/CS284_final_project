# coding=utf-8
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

from utils import *
import constants

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

def main():
    # Prepare devices
    device = constants.DISTILL_CONFIG['DEVICE']
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(DISTILL_CONFIG['SEED'])
    np.random.seed(DISTILL_CONFIG['SEED'])
    torch.manual_seed(DISTILL_CONFIG['SEED'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(DISTILL_CONFIG['SEED'])

    # Prepare task settings
    if os.path.exists(DISTILL_CONFIG['OUTPUT_DIR']) and os.listdir(DISTILL_CONFIG['OUTPUT_DIR']):
        raise ValueError(f'Output directory ({DISTILL_CONFIG["OUTPUT_DIR"]}) already exists and is not empty.')
    if not os.path.exists(DISTILL_CONFIG['OUTPUT_DIR']):
        os.makedirs(DISTILL_CONFIG['OUTPUT_DIR'])

    tokenizer = BertTokenizer.from_pretrained(DISTILL_CONFIG['STUDENT_MODEL'])

    train_dataloader = create_reliable_news_dataloader(
        os.path.join(DISTILL_CONFIG['FILE_PATH'], 'train.jsonl'),
        tokenizer,
        max_len = DISTILL_CONFIG['MAX_LEN'],
        batch_size = DISTILL_CONFIG['BATCH_SIZE'],
        shuffle=True,
        sample = DISTILL_CONFIG['SAMPLE'],
        title_only = DISTILL_CONFIG['TITLE_ONLY']
    )

    eval_dataloader = create_reliable_news_dataloader(
        os.path.join(DISTILL_CONFIG['FILE_PATH'], 'val.jsonl'),
        tokenizer,
        max_len = DISTILL_CONFIG['MAX_LEN'],
        batch_size = DISTILL_CONFIG['BATCH_SIZE'],
        sample = DISTILL_CONFIG['SAMPLE'],
        title_only = DISTILL_CONFIG['TITLE_ONLY']
    )

    teacher_model = TinyBertForSequenceClassification.from_pretrained(DISTILL_CONFIG['STUDENT_MODEL'], num_labels=1)
    teacher_model.to(device)

    student_model = TinyBertForSequenceClassification.from_pretrained(DISTILL_CONFIG['STUDENT_MODEL'], num_labels=1)
    student_model.to(device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader) * DISTILL_CONFIG['BATCH_SIZE'])
    logger.info("  Batch size = %d", DISTILL_CONFIG['BATCH_SIZE'])

    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    size = 0
    for n, p in student_model.named_parameters():
        logger.info('n: {}'.format(n))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    if not DISTILL_CONFIG['PRED_DISTILL']:
        schedule = 'none'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=DISTILL_CONFIG['LR'],
                            warmup=DISTILL_CONFIG['WARMUP'],
                            t_total= len(train_dataloader) * DISTILL_CONFIG['BATCH_SIZE'])
    # Prepare loss functions
    loss_mse = MSELoss()

    # Train and evaluate
    global_step = 0
    best_dev_acc = 0.0
    output_eval_file = os.path.join(DISTILL_CONFIG['OUTPUT_DIR'], "eval_results.txt")

    # Training loop
    for epoch_ in trange(int(DISTILL_CONFIG['EPOCHS']), desc="Epoch"):
        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.

        student_model.train()
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train

        # Evaluation
        student_model.eval()

        loss = tr_loss / (epoch + 1)
        cls_loss = tr_cls_loss / (epoch + 1)
        att_loss = tr_att_loss / (step + 1)
        rep_loss = tr_rep_loss / (step + 1)

        result = {}
        if DISTILL_CONFIG['PRED_DISTILL']:
            result = do_eval(student_model, task_name, eval_dataloader,
                                device, output_mode, eval_labels, num_labels)
        result['global_step'] = global_step
        result['cls_loss'] = cls_loss
        result['att_loss'] = att_loss
        result['rep_loss'] = rep_loss
        result['loss'] = loss

        result_to_file(result, output_eval_file)

        if not DISTILL_CONFIG['PRED_DISTILL']:
            save_model = True
        else:
            save_model = False
            if result['acc'] > best_dev_acc:
                best_dev_acc = result['acc']
                save_model = True

        if save_model:
            logger.info("***** Save model *****")

            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

            model_name = WEIGHTS_NAME
            output_model_file = os.path.join(DISTILL_CONFIG['OUTPUT_DIR'], model_name)
            output_config_file = os.path.join(DISTILL_CONFIG['OUTPUT_DIR'], CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(DISTILL_CONFIG['OUTPUT_DIR'])

if __name__ == "__main__":
    main()
