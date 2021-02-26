import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import wandb

from wandb.keras import WandbCallback
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, HoltWintersResults
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataset import DataSet
from models import *

"""
Cell with hyperparameters and arguments
The contents of this cell will probably have to be specified in the command line in the end
"""


"""
Weights and bias initialization and configuration
"""

parameters = {
        'dataset': 'sunspots',
        'multivariate': False,
        'gen_length': 48,
        'gen_batch': 2,
        'gen_features': 1,
        'epochs': 20,            # not that important with early stopping, just make sure it's high enough
        'test_val': 'val',
        'lstm_units': 50,
        'models': ['simple', 'simple_lstm', 'stacked_lstm'],
        'patience': 2,
        'optim': 'adam',
        'loss': 'mse',
        'repeats': 1
    }

# Dictionary with available datasets and models

DATASETS = {'sunspots': {'DATA_PATH': 'sunspots',
                         'TARGET': 'Sunspots',
                         'TIME_COLS': ['Month']},
            'pm2_5': {'DATA_PATH': 'pm2_5',
                      'TARGET': 'pm2.5',
                      'TIME_COLS': ['year', 'month', 'day', 'hour']}}

MODELS = {'naive': Naive(),
          'simple': Baseline(input_shape=(parameters['gen_batch'], parameters['gen_length'], parameters['gen_features'])),
          'simple_lstm': Simple_LSTM(input_shape=(parameters['gen_batch'], parameters['gen_length'],
                                                  parameters['gen_features']), units=parameters['lstm_units']),
          'stacked_lstm': Stacked_LSTM(input_shape=(parameters['gen_batch'], parameters['gen_length'],
                                                    parameters['gen_features']), units=parameters['lstm_units'])}

def get_batch(batch_nr):
    """
    Function to get batch
    """

    # initialize current batch with zeros
    batch_size = parameters['gen_batch']
    current_batch = np.zeros((batch_size, parameters['gen_length'], parameters['gen_features']))

    # fill in each row of batch
    for batch in range(batch_size):
        start_idx = parameters['gen_length'] - (batch + batch_nr*batch_size)

        # if there is still data left from training set, take that
        if start_idx > 0:
            current_batch[batch, :start_idx, :] = train_X[-start_idx:]
            current_batch[batch, start_idx:, :] = val_X[:batch + batch_nr*batch_size]
        else: # only get data from validation data
            current_batch[batch] = val_X[np.abs(start_idx): batch+batch_nr*batch_size]

    return current_batch

if __name__ == "__main__":

    # Create own DataSet object and perform preprocessing
    df = DataSet(parameters['multivariate'], DATASETS[parameters['dataset']])
    df.preprocess(parameters['test_val'])

    # Get loaders
    train_generator = df.get_loaders(split='train', length=parameters['gen_length'], batch_size=parameters['gen_batch'])
    test_generator = df.get_loaders(split=parameters['test_val'], length=parameters['gen_length'], batch_size=parameters['gen_batch'])

    train_X, train_y = df.get_data('train')
    val_X, val_y = df.get_data(parameters['test_val'])

    assert len(val_y) % parameters['gen_batch'] == 0, f'Batch size of {parameters["gen_batch"]} is not compatible with' \
                                                      f'length of dataset {len(val_y)}'
    # print(f"Test generator first sample: {test_generator[0]}")
    # print(f"First val data: {val_y[0]}\n"
    #       f"Second val data: {val_y[1]}\n"
    #       f"")

    transformer = df.get_transformer('y')
    inv_true_vals = np.squeeze(transformer.inverse_transform(val_y))

    # Loop over every model
    for curr_model in parameters['models']:

        experiment_name = f'{parameters["dataset"]}_{curr_model}'

        for _ in range(parameters['repeats']):

            print(f"Repeat nr {_} of {curr_model}")

            # Initialize new run for every model
            run = wandb.init(
                            project='Master thesis',
                            group=experiment_name,
                            config=parameters,
                            reinit=True
            )

            model = MODELS[curr_model]
            model.compile(optimizer=parameters['optim'], loss=parameters['loss'],
                          metrics=[tf.keras.metrics.RootMeanSquaredError()])

            if curr_model != 'naive':       # naive model does not need training
                model.fit(train_generator, epochs=parameters['epochs'], validation_data=test_generator,
                                    callbacks=[EarlyStopping(patience=parameters['patience']), WandbCallback()])

            # Create own test loop here, since test generator cannot predict on first value of val set
            predictions = []

            for batch_nr in range(int(len(val_y)/parameters['gen_batch'])):
                current_batch = get_batch(batch_nr)
                current_prediction = model.predict(current_batch)
                predictions.append(current_prediction)

            inv_predictions = np.squeeze(transformer.inverse_transform(np.reshape(predictions, (-1, 1))))

            error = mean_absolute_error(inv_true_vals, inv_predictions)
            rmse = np.sqrt(mean_squared_error(inv_true_vals, inv_predictions))

            run.log({'Model': curr_model,
                        'MAE': error,
                        'rMSE': rmse})

            run.finish()

            print((error, rmse))

