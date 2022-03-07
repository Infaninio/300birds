import argparse
import os
from typing import Dict
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def get_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--destination', help='Folder where the downloaded data should be stored', default='./dataset')
    return parser.parse_args()


def create_directories(path: str):
    storage_path = os.path.abspath(path)
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)
    training_file = os.path.abspath(__file__)
    training_file = os.path.join(training_file[:training_file.rfind('\\')], 'train.yaml')

    # Create train.yaml
    data: Dict[str, str] = {'repo_version': '0.0.1', 'dataset_path': storage_path}
    with open(training_file, mode='w+') as fp:
        yaml.dump(data, fp)
        fp.writelines(['\n', '# Configurations for the machine learning algorithm.'])


def download_routine(destination_folder: str):
    
    dataset = 'gpiosenka/100-bird-species'
    # Guide to get a api-key https://python.plainenglish.io/how-to-use-kaggle-api-in-python-7bddb3d87c15
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, destination_folder)
    with zipfile.ZipFile('./dataset/100-bird-species.zip') as z:
        z.extractall('./dataset')


if __name__ == '__main__':
    args = get_cla()
    create_directories(args.destination)
    download_routine(args.destination)