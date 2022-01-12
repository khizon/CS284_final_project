import os
import wandb

if __name__ == '__main__':
    # Dataset
    url = 'https://drive.google.com/drive/folders/11mRvsHAkggFEJvG4axH4mmWI6FHMQp7X?usp=sharing'
    data = 'data/nela_gt_2018_site_split'
    
    os.system(f'gdown --folder {url} -O {data}')
    
    # TinyBERT models (student)
    # url = 'https://drive.google.com/drive/folders/1aKj6nmFQxaVmmBdKvJ7dtcYdOWKEMP35?usp=sharing'
    # data = 'artifacts/'
    
    # os.system(f'gdown --folder {url} -O {data}')
    # os.system(f'unzip {data}/TinyBERT/TinyBERT_4L_312D.zip -d {data}/')
    # os.system(f'unzip {data}/TinyBERT/TinyBERT_6L_768D.zip -d {data}/')
    
    # Bert-title-content (teacher)
    # run = wandb.init()
    # artifact = run.use_artifact('khizon/UnreliableNews/BERT-title-content-benchmark:v0', type='model')
    # artifact_dir = artifact.download()
    # run.finish()
    
    # os.rename(os.path.join('artifacts', 'BERT-title-content-benchmark:v0', 'torch_checkpoint.bin'),
    #           os.path.join('artifacts', 'BERT-title-content-benchmark:v0', 'pytorch_model.bin'))
    