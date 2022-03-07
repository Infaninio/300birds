import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(kwargs)
        self.rescale = tf.keras.layers.Rescaling(1./255)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.max_pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(256, activation='relu')
        self.predictions = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = self.rescale(inputs)
        inputs = self.conv1(inputs)
        inputs = self.max_pool1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.max_pool2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.max_pool3(inputs)
        inputs = self.flatten(inputs)
        inputs = self.dense(inputs)
        return self.predictions(inputs)