import sys
import os

from text_processing_util import TextProcessing
from text_cnn import kimCNN
import cPickle


MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2




pos_file = 'rt-polarity.pos'
neg_file = 'rt-polarity.neg'
fname = sys.argv[1]


# Prepare text samples and their labels
print('Processing text dataset')

labels = []
texts = []
with open(pos_file, "rb") as f:
	for line in f:
	    labels.append('pos')
	    texts.append(line.strip())


with open(neg_file, "rb") as f:
	for line in f:
	    labels.append('neg')
	    texts.append(line.strip())


print("Found %s texts" %len(texts))
print("Found %s labels" %len(labels))




tp = TextProcessing(texts, labels, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT)

x_train, y_train, x_val, y_val, word_index = tp.preprocess()
embeddings_index = tp.build_embedding_index_from_word2vec(fname, word_index)
print('Found %s word vectors.' % len(embeddings_index))

cPickle.dump([word_index, embeddings_index], open('tokenization_and_embedding.p', 'wb'))

labels_index = tp.labels_index

model = kimCNN(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, embeddings_index, word_index, labels_index=labels_index)
print(model.summary())

model.fit(x=x_train, y=y_train, batch_size=50, epochs=25 , validation_data=(x_val, y_val))
# model.save('test_model.h5')









