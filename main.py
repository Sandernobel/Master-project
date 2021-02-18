import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from dataset import DataSet
from models import Baseline

"""
Cell with hyperparameters and arguments
The contents of this cell will probably have to be specified in the command line in the end
"""

# Dataset hyperparameters
DATA_PATH = 'sunspots'
TARGET = 'Sunspots'
TIME_COLS = ['Month']
MULTIVARIATE = False

# Time series generator parameters
LENGTH = 2
BATCH_SIZE = 1

# Training parameters
EPOCHS = 10

def seed_everything(seed=1):
    """
    Function to set random seeds for reproducibility
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    seed_everything()

    # Create own DataSet object and perform preprocessing
    df = DataSet(DATA_PATH, TIME_COLS, TARGET, MULTIVARIATE)
    df.preprocess()

    # Get loaders
    train_generator = df.get_loaders(split='train', length=LENGTH, batch_size=BATCH_SIZE)
    val_generator = df.get_loaders(split='val', length=LENGTH, batch_size=BATCH_SIZE)

    x1, y1 = train_generator[0]
    n_features = x1.shape[1]
    print(f"Samples (generator length): \t {len(train_generator)}\n"
          f"Sample 1 shape: \t {x1.shape, y1.shape}\n"
          f"Sample 1: \t {x1, y1}\n")

    model = Baseline(input_shape=(BATCH_SIZE, LENGTH))
    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(generator=train_generator, epochs=EPOCHS)

    train_vals = df.get_data('train')[1]
    predictions = model.predict_generator(generator=val_generator)
    true_vals = df.get_data('val')[1]

    transformer = df.get_transformer('y')

    inv_train_vals = np.squeeze(transformer.inverse_transform(train_vals))
    inv_predictions = np.squeeze(transformer.inverse_transform(predictions))
    inv_true_vals = np.squeeze(transformer.inverse_transform(true_vals))[LENGTH:]

    plt.plot(inv_train_vals, label='Train')
    plt.plot([None for i in train_vals] + [i for i in inv_predictions], label='Predictions')
    plt.plot([None for i in train_vals] + [i for i in inv_true_vals], label='True')
    # print(model.history.history)
    # losses = list(model.history.history.values())[0]
    # print(losses)
    # plt.plot(losses)
    plt.legend()
    plt.show()