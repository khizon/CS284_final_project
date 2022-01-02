from tqdm.auto import tqdm

import torch

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

from utils import *
from constants import *
import wandb

def task_distill(config = None):
    with wandb.init(config=config, entity=FILES['USER']) as run:
        config = wandb.config
        seed_everything(config.seed)

        # Download Path of Teacher Model weights
        user = FILES['USER']
        project = FILES['PROJECT']
        artifact_ver = f'{FILES["MODEL_NAME"]}:{FILES["VERSION"]}'
        model_path = f"{user}/{project}/{artifact_ver}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        # Initialize student and teacher models
        tokenizer = BTokenizer.from_pretrained('bert-base-cased')
        student = TinyBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 1)
        teacher = TinyBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 1)

        # Get the teacher model weights from WandB
        artifact = run.use_artifact(model_path, type = 'model')
        artifact_dir = artifact.download()
        checkpoint = torch.load(os.path.join('artifacts', artifact_ver, 'torch_checkpoint.bin'))
        teacher.load_state_dict(checkpoint['state_dict'])

        # Get the student model from the General Tiny BERTs
        checkpoint = torch.load(os.path.join('generalTinyBERT', config.tinybert, 'pytorch_model.bin'))

        teacher.to(device)
        student.to(device)

        # Initialize Train and Eval data set
        train_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'train.jsonl'),
            tokenizer,
            max_len = config.max_len,
            batch_size = config.batch_size * max(1, n_gpu),
            shuffle=True,
            sample = config.sample,
            title_only = config.title_only
        )

        val_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'val.jsonl'),
            tokenizer,
            max_len = config.max_len,
            batch_size = config.batch_size * max(1, n_gpu),
            sample = config.sample,
            title_only = config.title_only
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        # Optimizer and Scheduler
        schedule = 'warmup_linear' if not config.pred_distill else 'none'
        total_steps = len(train_data_loader) * config.epochs
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=config.learning_rate,
                             warmup=int(total_steps * config.warmup),
                             t_total=total_steps)

        # Initialize early stopping
        best_accuracy = 0.0
        early_stopping = EarlyStopping(patience = config.patience, min_delta = config.min_delta)

        for epoch in range(config.epochs):
            print(f'Distillation {epoch + 1}/{config.epochs}:')

            train_acc, att_loss, rep_loss = distill_train_epoch(student, teacher, train_data_loader, optimizer, device)

            val_acc, val_loss = eval_model(student, 'bert-base-cased', val_data_loader, device)

            wandb.log({
                'train acc' : train_acc,
                'att loss' : att_loss,
                'rep loss' : rep_loss,
                'val_acc' : val_acc,
                'val_loss' : val_loss
            })

            # Checkpoint Best Model
            if val_acc > best_accuracy:
                checkpoint = {
                    'state_dict' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss' : val_loss,
                    'accuracy': val_acc,
                    'epoch': epoch
                }
                
                if not os.path.exists(os.path.join('checkpoint')):
                    os.makedirs(os.path.join('checkpoint'))
                torch.save(checkpoint, os.path.join('checkpoint', 'torch_checkpoint.bin'))
                best_accuracy = val_acc
            
            #Stop training when accuracy plateus.
            early_stopping(val_acc)
            if early_stopping.early_stop:
                break

        artifact = wandb.Artifact(FILES['MODEL_NAME'], type = 'model')
        artifact.add_file(os.path.join('checkpoint', 'torch_checkpoint.bin'))
        run.log_artifact(artifact)
        run.join()
        run.finish()

if __name__ == '__main__':
    transformers.logging.set_verbosity_info()
    sweep_id = wandb.sweep(sweep_config, project = FILES['PROJECT'])
    wandb.agent(sweep_id, task_distill, count=1)