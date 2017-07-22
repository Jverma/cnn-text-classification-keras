from __future__ import print_function

import numpy as np

from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras import optimizers
from keras import regularizers

import sys
from text_processing_util import TextProcessing



__author__ = 'jverma'



## sentence CNN by Y.Kim
def kimCNN(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, embeddings_index, word_index, labels_index):
    """
    Convolution neural network model for sentence classification.

    Parameters
    ----------
    EMBEDDING_DIM: Dimension of the embedding space.
    MAX_SEQUENCE_LENGTH: Maximum length of the sentence.
    MAX_NB_WORDS: Maximum number of words in the vocabulary.
    embeddings_index: A dict containing words and their embeddings.
    word_index: A dict containing words and their indices.
    labels_index: A dict containing the labels and their indices.

    Returns
    -------
    compiled keras model
    """
    print('Preparing embedding matrix.')
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)


    print('Training model.')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    print(embedded_sequences.shape)


    # add first conv filter
    embedded_sequences = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPooling2D((MAX_SEQUENCE_LENGTH - 5 + 1, 1))(x)


    # add second conv filter.
    y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPooling2D((MAX_SEQUENCE_LENGTH - 4 + 1, 1))(y)


    # add third conv filter.
    z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPooling2D((MAX_SEQUENCE_LENGTH - 3 + 1, 1))(z)


    # concate the conv layers
    alpha = concatenate([x,y,z])

    # flatted the pooled features.
    alpha = Flatten()(alpha)

    # dropout
    alpha = Dropout(0.5)(alpha)

    # predictions
    preds = Dense(len(labels_index), activation='softmax')(alpha)

    # build model
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta()
        
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])


    return model
