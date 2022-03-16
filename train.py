import os
import hydra
from omegaconf import DictConfig
from src.load_dataset import load_dataset, DatasetType
import tensorflow as tf
import src.model

@hydra.main(config_path="./", config_name="train.yaml")
def train(cfg: DictConfig):
    
    train_data = load_dataset(cfg.dataset_path, type=DatasetType.TRAIN, batch_size=cfg.batch_size, image_size=tuple(cfg.image_size))
    validation_data = load_dataset(cfg.dataset_path, type=DatasetType.VALIDATION, batch_size=cfg.batch_size, image_size=tuple(cfg.image_size))
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model: tf.keras.Model = hydra.utils.instantiate(cfg.model)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=metrics)
    model.fit(x=train_data, validation_data=validation_data, epochs=20, use_multiprocessing=True, workers=8)


if __name__ == "__main__":
    print(os.curdir)
    train()