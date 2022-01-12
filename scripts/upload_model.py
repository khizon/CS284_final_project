import wandb
from transformers import DistilBertForSequenceClassification
import os
import torch


if __name__ == '__main__':
    # with wandb.init() as run:
    #     artifact = run.use_artifact('khizon/UnreliableNews/distilBERT-title-benchmark:v2', type='model')
    #     artifact_dir = artifact.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', dropout = 0.1, num_labels = 2, n_layers = 6)
    checkpoint = torch.load(os.path.join('artifacts', 'distilBERT-title-benchmark-v2', 'pytorch_model.bin'), map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    model.push_to_hub("distilbert-unreliable-news-eng")