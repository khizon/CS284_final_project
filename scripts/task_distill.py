from tqdm.auto import tqdm

import torch

from transformer.modeling import TinyBertForSequenceClassification, BertConfig as TinyBertConfig
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

import transformers

from utils import *
from constants import *
import wandb

def task_distill(config = None):
    with wandb.init(config=config) as run:
        config = wandb.config
        seed_everything(config.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        # Initialize Tokenizer and Teacher Model
        tokenizer, teacher = tokenizer, model = create_model('bert-base-cased', distill = True)
        checkpoint = torch.load(os.path.join('artifacts', config.teacher_model, 'pytorch_model.bin'), map_location=torch.device(device))
        teacher.load_state_dict(checkpoint['state_dict'])

        # Initialize Student Model (General TinyBert)
        student_path = os.path.join('artifacts', config.student_model)
        student = TinyBertForSequenceClassification.from_pretrained(student_path, num_labels = 1)
        print('Student Model')
        print(student.config.to_dict())

        if not os.path.exists(os.path.join('artifacts', 'temp')):
            os.makedirs(os.path.join('artifacts', 'temp'))
        
        with open(os.path.join('artifacts', 'temp', 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(student.config.to_dict(), f, ensure_ascii=False, indent=4)

        if n_gpu > 1:
            teacher = torch.nn.DataParallel(teacher)
            student = torch.nn.DataParallel(student)

        teacher.to(device)
        student.to(device)

        # Initialize Train and Eval data set
        train_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'train.jsonl'),
            tokenizer,
            max_len = 512,
            batch_size = config.batch_size * max(1, n_gpu),
            shuffle=True,
            sample = config.sample,
            title_only = False
        )

        val_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'val.jsonl'),
            tokenizer,
            max_len = 512,
            batch_size = config.batch_size * max(1, n_gpu),
            sample = config.sample,
            title_only = False
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
        optimizer = AdamW(optimizer_grouped_parameters, lr = config.learning_rate)
        total_steps = len(train_data_loader) * config.epochs
        if config.pred_distill:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps =  int(total_steps * config.warmup),
                num_training_steps = total_steps
            )

        # Initialize early stopping
        best_accuracy = 0.0
        early_stopping = EarlyStopping(patience = config.patience, min_delta = config.min_delta)

        for epoch in range(config.epochs):
            print(f'Distillation {epoch + 1}/{config.epochs}:')

            train_acc, train_loss = distill_train_epoch(student, teacher, train_data_loader, optimizer, device, config.pred_distill)

            val_acc, val_loss = eval_model(student, 'tiny-bert', val_data_loader, device)

            wandb.log({
                "train acc": train_acc,
                "train_loss": train_loss,
                "val acc": val_acc,
                "val_loss": val_loss,
                "epoch" : epoch
            })

            
            # If pred_distill Checkpoint Best Model else checkpoint every 5 epochs
            if (config.pred_distill and (val_acc > best_accuracy)) or ((not config.pred_distill) and ((epoch+1)%5 == 0)):
                checkpoint = {
                    'state_dict' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss' : val_loss,
                    'accuracy': val_acc,
                    'epoch': epoch
                }
                print('***Saving Checkpoint***')
                torch.save(checkpoint, os.path.join('artifacts', 'temp', 'pytorch_model.bin'))
                best_accuracy = val_acc
            
            #Stop training when accuracy plateus (only on pred distil).
            if config.pred_distill:
                early_stopping(val_acc)
                if early_stopping.early_stop:
                    break
            
                    
                    
        

        # Testing
        model = TinyBertForSequenceClassification.from_pretrained(os.path.join('artifacts', 'temp'), num_labels = 1)
        # checkpoint = torch.load(os.path.join('artifacts', 'pytorch_model.bin'),map_location=torch.device(device))
        # model.load_state_dict(checkpoint['state_dict'])

        model.to(device)

        # Initialize test data set
        test_data_loader = create_reliable_news_dataloader(
            os.path.join(config.dataset_path, 'test.jsonl'),
            tokenizer,
            max_len = 512,
            batch_size = 8 * max(1, n_gpu),
            sample = config.sample,
            title_only = False
        )
        
        y_pred, y_test, test_acc, ave_time = get_predictions(model, 'tiny-bert', test_data_loader, device)

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

        artifact = wandb.Artifact(FILES['MODEL_NAME'], type = 'model')
        artifact.add_file(os.path.join('artifacts', 'temp', 'pytorch_model.bin'))
        artifact.add_file(os.path.join('artifacts', 'temp', 'config.json'))
        artifact.add_file(os.path.join('artifacts', 'temp', 'test_results.json'))
        run.log_artifact(artifact)
        run.join()
        run.finish()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    transformers.logging.set_verbosity_info()
    sweep_config['parameters'] = distill_dict
    sweep_id = wandb.sweep(sweep_config, project = FILES['PROJECT'])
    wandb.agent(sweep_id, task_distill, count=1)