import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

_DEFAULT_EMBEDDING_DIM = 128


class KerasLstmModelBuilder:

    def __init__(self):
        pass

    def build_lstm_with_embedding_model(self, max_num_words, max_seq_len):
        sequence_input = Input(shape=(max_seq_len,), dtype='int32')

        embedded_seq = Embedding(max_num_words, _DEFAULT_EMBEDDING_DIM)(sequence_input)
        lstm_seq = LSTM(_DEFAULT_EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5)(embedded_seq)
        output = Dense(1, activation='sigmoid')(lstm_seq)

        model = Model(sequence_input, output)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


