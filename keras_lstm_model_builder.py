import os
import sys
import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

_DEFAULT_GLOVE_DATA_FILE = "data/glove.6B.100d.txt"
_DEFAULT_EMBEDDING_DIM = 128


class KerasLstmModelBuilder:

    def __init__(self):
        self._embedding_index = {}
        with open(_DEFAULT_GLOVE_DATA_FILE) as glove_data:
            for word_vec in glove_data:
                values = word_vec.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                self._embedding_index[word] = vec

        print("Loaded {} word vectors from GloVe".format(len(self._embedding_index)))


    def build_lstm_with_embedding_model(self, max_num_words, max_seq_len):
        sequence_input = Input(shape=(max_seq_len,), dtype='int32')

        # prepare embedding matrix
        embedding_num_words = min(max_num_words, len(self._embedding_index) + 1)
        embedding_matrix = np.zeros((embedding_num_words, _DEFAULT_EMBEDDING_DIM))
        for word, i in self._embedding_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = self._embedding_index.get(word)
            if embedding_vector is None:
                # words not found in embedding index will be all-zeros.
                pass
            else:
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(embedding_num_words, _DEFAULT_EMBEDDING_DIM, weights=[embedding_matrix],
                                    input_length=max_seq_len, trainable=False)

        embedded_seq = embedding_layer(sequence_input)
        lstm_seq = LSTM(_DEFAULT_EMBEDDING_DIM)
        output = Dense(1, activation='sigmoid')(lstm_seq)

        model = Model(sequence_input, output)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


