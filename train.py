import hydra
from omegaconf import DictConfig
from src.load_dataset import load_dataset, DatasetType
import tensorflow as tf

@hydra.main(config_path="./", config_name="train.yaml")
def train(cfg: DictConfig):
    model: tf.keras.Model = hydra.utils.instantiate(cfg.model)
    train_data = load_dataset(cfg.dataset_path, type=DatasetType.TRAIN, batch_size=32, image_size=(132, 132))
    print(train_data.class_names)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=metrics)
    model.fit(train_data, epochs=6)


if __name__ == "__main__":
    train()