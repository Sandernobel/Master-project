import tensorflow as tf
import tensorflow.keras.layers as l

class Baseline(tf.keras.Model):
    """
    Toy ANN for practice
    """
    def __init__(self, input_shape):
        super().__init__()
        self.layer = l.Dense(1, input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        x = self.layer(x)
        return x

class Naive(tf.keras.Model):
    """
    Returns y_hat = y_t-1
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs, training=None, mask=None):
        return inputs[:, -1, -1]

class Simple_LSTM(tf.keras.Model):
    """
    LSTM
    """
    def __init__(self, input_shape, units=100):
        super().__init__()
        self.lstm = l.LSTM(units, activation='relu', input_shape=input_shape, return_sequences=False)
        self.dense = l.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.lstm(inputs)
        return self.dense(x)

class Stacked_LSTM(tf.keras.Model):
    """
    LSTM
    """
    def __init__(self, input_shape, units=100):
        super().__init__()
        self.lstm1 = l.LSTM(units, activation='relu', input_shape=input_shape, return_sequences=True)
        self.lstm2 = l.LSTM(units, activation='relu', input_shape=input_shape, return_sequences=False)
        self.dense = l.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return self.dense(x)