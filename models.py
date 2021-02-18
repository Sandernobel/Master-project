import tensorflow as tf
import tensorflow.keras.layers as l

class Baseline(tf.keras.Model):
    """
    Toy ANN for practice
    """
    def __init__(self, input_shape):
        super().__init__()
        self.layer = l.Dense(1, input_shape=input_shape)

    def call(self, inputs):
        return self.layer(inputs)
