# CS284_final_project

## To install dependencies
> `pip install -r requirements.txt`

## Downloading the dataset
The original dataset is fromt the [NELA 2018](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ULHLCB) dataset, by [Noregaard et. al](https://arxiv.org/abs/1904.01546). For this project, instead of randomly splitting entries into train, validation, and test sets, they are split by source as described in this paper by [Zhou et. al](https://arxiv.org/pdf/2104.10130v1.pdf).
The author's splitting script can be found [here](https://owenzx.github.io/unreliable_news). The split that I had used for this project is stored in Google Drive and can be downloaded using the script below.
> `python scripts/download_dataset.py`
This will download the train, validation and test sets in `data/nela-gt-2018-site-split/`

## Fine-tuning benchmark models
Open `scripts/constants.py` and modify the `parameters_dict`.
Supported models from the HuggingFace model hub are `bert-base-cased`, `distilbert-base-cased`, and `google/mobilebert-uncased`
Run the command:
> `python scripts/train_benchmark.py`
If you have a Weights and Biases account, you can choose option **2** when prompted, and enter your API key to log training progress online.
The checkpoint will be saved in `artifacts/temp`


## Released models on the HuggingFace ModelHub
- TBA

