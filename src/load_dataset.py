import os
from typing import Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
from enum import Enum

class DatasetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


def load_dataset(dataset_path: str, type: DatasetType, batch_size: int, image_size: Tuple[int]) -> tf.data.Dataset:
    if type == DatasetType.TRAIN:
        return tf.keras.utils.image_dataset_from_directory(os.path.join(dataset_path, 'train'), batch_size=batch_size, image_size=image_size)
    if type == DatasetType.VALIDATION:
        return tf.keras.utils.image_dataset_from_directory(os.path.join(dataset_path, 'valid'), batch_size=batch_size, image_size=image_size)
    if type == DatasetType.TEST:
        return tf.keras.utils.image_dataset_from_directory(os.path.join(dataset_path, 'test'), batch_size=batch_size, image_size=image_size)



if __name__ == '__main__':
    test = load_dataset('C:\\Users\\marti\\source\\repos\\Machine Learning\\mnist\\dataset\\train-images-idx3-ubyte.dat',
                 'C:\\Users\\marti\\source\\repos\\Machine Learning\\mnist\\dataset\\train-labels-idx1-ubyte.dat')