import os

if __name__ == '__main__':
    url = 'https://drive.google.com/drive/folders/11mRvsHAkggFEJvG4axH4mmWI6FHMQp7X?usp=sharing'
    data = 'data/nela_gt_2018_site_split'

    os.system(f'gdown --folder {url} -O {data}')