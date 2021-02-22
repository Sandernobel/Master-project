import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, HoltWintersResults
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataset import DataSet
from models import *

"""
Cell with hyperparameters and arguments
The contents of this cell will probably have to be specified in the command line in the end
"""

# Dataset hyperparameters

DATASETS = {'sunspots': {'DATA_PATH': 'sunspots',
                         'TARGET': 'Sunspots',
                         'TIME_COLS': ['Month']},
            'pm2_5': {'DATA_PATH': 'pm2_5',
                      'TARGET': 'pm2.5',
                      'TIME_COLS': ['year', 'month', 'day', 'hour']}}

CURR_DATASET = 'sunspots'
MULTIVARIATE = False

# Time series generator parameters
LENGTH = 24
BATCH_SIZE = 4
FEATURES = 1

# Training parameters
EPOCHS = 20             # not that important with early stopping, just make sure it's high enough
TEST_VAL = 'val'


# Models
MODELS = {'naive': Naive(),
          'simple': Baseline(input_shape=(BATCH_SIZE, LENGTH, FEATURES)),
          'simple_lstm': Simple_LSTM(input_shape=(BATCH_SIZE, LENGTH, FEATURES))}
CURR_MODELS = ['naive', 'simple_lstm']

# Plotting parameters
TIME = 100

def seed_everything(seed=1):
    """
    Function to set random seeds for reproducibility
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    seed_everything()

    # Create own DataSet object and perform preprocessing
    df = DataSet(MULTIVARIATE, DATASETS[CURR_DATASET])
    df.preprocess()

    # Get loaders
    train_generator = df.get_loaders(split='train', length=LENGTH, batch_size=BATCH_SIZE)
    test_generator = df.get_loaders(split=TEST_VAL, length=LENGTH, batch_size=BATCH_SIZE)

    # Define early stopping
    early_stop = EarlyStopping(patience=2)

    for curr_model in CURR_MODELS:
        model = MODELS[curr_model]
        model.compile(optimizer='adam', loss='mse')

        if curr_model != 'naive':       # naive model does not need training
            model.fit_generator(generator=train_generator, epochs=EPOCHS, validation_data=test_generator,
                            callbacks=[early_stop])

        train_X, train_y = df.get_data('train')
        val_X, val_y = df.get_data(TEST_VAL)

        predictions = model.predict_generator(generator=test_generator)

        transformer = df.get_transformer('y')

        inv_predictions = np.squeeze(transformer.inverse_transform(np.reshape(predictions, (-1,1))))
        inv_true_vals = np.squeeze(transformer.inverse_transform(val_y))[LENGTH:]

        plt.plot(inv_predictions[-TIME:], label=f'Predictions {curr_model}')

        error = mean_absolute_error(inv_true_vals, inv_predictions)
        rmse = np.sqrt(mean_squared_error(inv_true_vals, inv_predictions))

        print((error, rmse))

    plt.plot(inv_true_vals[-TIME:], label='True')
    plt.legend()
    plt.show()