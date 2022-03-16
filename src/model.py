from typing import List, Tuple
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

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size: Tuple, filter: int, num_convolutions: int, stride_for_first: int = 1, learnable_skip_connection: bool = False) -> None:
        super().__init__()
        self.convolutions: List[tf.keras.layers.Layer] = []
        self.batch_norms: List[tf.keras.layers.Layer] = []
        self.activations: List[tf.keras.layers.Layer] = []
        self.learnable_skip_Connection: tf.keras.Layer = None
        if stride_for_first > 1:
            self.convolutions.append(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=(stride_for_first, stride_for_first), padding='same'))          
            self.batch_norms.append(tf.keras.layers.BatchNormalization())
            self.activations.append(tf.keras.layers.ReLU())
            num_convolutions -= 1
            if learnable_skip_connection:
                self.learnable_skip_Connection = tf.keras.layers.Conv2D(filters=filter, kernel_size=(1, 1), strides=(stride_for_first, stride_for_first))
            else:
                # TODO non learnable Skip Connection
                self.learnable_skip_Connection = tf.keras.layers.Conv2D(filters=filter, kernel_size=(1, 1), strides=(stride_for_first, stride_for_first))
        
        for _ in range(num_convolutions):
            self.batch_norms.append(tf.keras.layers.BatchNormalization())
            self.activations.append(tf.keras.layers.ReLU())
            self.convolutions.append(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, padding='same'))

    def call(self, inputs: tf.Tensor, **kwargs):
        skipValue = inputs
        x = inputs

        for cnt, layer in enumerate(self.convolutions):
            x = layer(x)
            x = self.batch_norms[cnt](x)
            if not cnt % 2 and cnt != 0:
                if cnt == 2 and self.learnable_skip_Connection:
                    x = x + self.learnable_skip_Connection(skipValue)
                else:
                    x = x + skipValue
                skipValue = x
            x = self.activations[cnt](x)
        return x


class ResNet34(tf.keras.Model):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))
        self.avg_pooling_start = tf.keras.layers.AveragePooling2D()
        self.conv_block1 = ConvLayer((3,3), 64, 6)
        self.conv_block2 = ConvLayer((3,3), 128, 8, 2, True)
        self.conv_block3 = ConvLayer((3,3), 256, 12, 2, True)
        self.conv_block4 = ConvLayer((3,3), 512, 6, 2, True)
        self.avg_pooling_end = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes,  activation='softmax')
    
    def call(self, inputs: tf.Tensor, **kwargs):
        x = self.conv1(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = self.avg_pooling_start(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avg_pooling_end(x)
        x = self.flatten(x)
        return self.dense(x)