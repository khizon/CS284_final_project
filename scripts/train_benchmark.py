from tqdm.auto import tqdm

import torch

import transformers
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

from utils import *
from constants import *
import wandb

def train(config = None):
    
    with wandb.init(config=config, entity=FILES['USER']) as run:
        config = wandb.config
        seed_everything(config.seed)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        # Initialize Model
        tokenizer, model = create_model(config.model_name, config.dropout, config.freeze_bert)
        
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

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
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        # Optimizer and Scheduler
        optimizer = AdamW(optimizer_grouped_parameters, lr = config.learning_rate)
        total_steps = len(train_data_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps =  int(total_steps * config.warmup),
            num_training_steps = total_steps
        )

        # Mixed precision Gradient Scaler
        # scaler = torch.cuda.amp.GradScaler()
        scaler = None

        # Save Model Config
        if not os.path.exists(os.path.join('artifacts')):
                os.makedirs(os.path.join('artifacts'))
            
        with open(os.path.join('artifacts', 'temp', 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=4)

        # Initialize Early Stopping
        best_accuracy = 0.0
        early_stopping = EarlyStopping(patience = config.patience, min_delta = config.min_delta)

        # Training Loop
        wandb.watch(model, log='all')
        for epoch in range(config.epochs):
            print(f'Training {epoch + 1}/{config.epochs}:')

            train_acc, train_loss = train_epoch(
                model, config.model_name, train_data_loader, optimizer, 
                device, scheduler,scaler
            )

            val_acc, val_loss = eval_model(model, config.model_name, val_data_loader, device)

            wandb.log({
                "train acc": train_acc,
                "train_loss": train_loss,
                "val acc": val_acc,
                "val_loss": val_loss,
                "epoch" : epoch
            })

            # Checkpoint Best Model
            if val_acc > best_accuracy:
                checkpoint = {
                    'state_dict' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'loss' : val_loss,
                    'accuracy': val_acc,
                    'epoch': epoch
                }
                
                # model.save_pretrained(os.path.join('checkpoint'))
                torch.save(checkpoint, os.path.join('artifacts', 'temp', 'pytorch_model.bin'))
                best_accuracy = val_acc
            
            #Stop training when accuracy plateus.
            early_stopping(val_acc)
            if early_stopping.early_stop:
                break
        
        # Testing
        
        # Initialize Tokenizer and Model
        tokenizer, model = create_model(config.model_name, config.dropout)
        
        # Get weights of best model
        checkpoint = torch.load(os.path.join('artifacts', 'temp', 'pytorch_model.bin'))
        model.load_state_dict(checkpoint['state_dict'])
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
        
        # Initialize test data set
        test_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'test.jsonl'),
            tokenizer,
            max_len = config.max_len,
            batch_size = config.batch_size * max(1, n_gpu),
            sample = config.sample,
            title_only = config.title_only
        )
        
        y_pred, y_test, test_acc, ave_time = get_predictions(model, config.model_name, test_data_loader, device)

        test_results = {
            'predictions': y_pred,
            'labels': y_test,
            'test_acc' : test_acc,
            'ave_time' : ave_time
        }

        wandb.log({
            "test acc": test_acc,
            "ave_time": ave_time
        })

        with open(os.path.join('artifacts', 'temp', 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        # Save model to weights and biases
        artifact = wandb.Artifact(FILES['MODEL_NAME'], type='model')
        artifact.add_file(os.path.join('artifacts', 'temp', 'pytorch_model.bin'))
        artifact.add_file(os.path.join('artifacts', 'temp', 'config.json'))
        artifact.add_file(os.path.join('artifacts', 'temp', 'test_results.json'))

        run.log_artifact(artifact)
        run.join()
        run.finish()

if __name__ == '__main__':
    transformers.logging.set_verbosity_info()
    sweep_config['parameters'] = parameter_dict
    sweep_id = wandb.sweep(sweep_config, project = FILES['PROJECT'])
    wandb.agent(sweep_id, train, count=1)