import argparse

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

from keras_lstm_model_builder import KerasLstmModelBuilder


def train_lstm_model_with_imdb_review(batch_size, epoches):
    lstm_model_builder = KerasLstmModelBuilder()


parser = argparse.ArgumentParser(description="Training LSTM Model for Sentiment Analysis with Keras")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()
    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Training the lstm model
    train_lstm_model_with_imdb_review(batch_size, epochs)

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()