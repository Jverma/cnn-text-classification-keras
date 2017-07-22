# cnn-text-classification-keras
Convolutional Neural Network for Text Classification in Keras


This is a Keras implementation of Yoon Kim's paper [Convolution Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) with the addition that this code also works for the [Glove vectors](https://nlp.stanford.edu/projects/glove/) and [Fasttext vectors](https://github.com/facebookresearch/fastText).

## Requirements:
- numpy
- keras
- cPickle


## Usage:
- Download the pre-trained Google `word2vec` word embedding vectors as a binary file from [here](https://code.google.com/p/word2vec/)


- Pre-process the text data
```
from text_processing_util import TextProcessing

tp = TextProcessing(texts, labels, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT)
```
where
	- texts: a list of sentences.
	- labels: a list of labels corresponding to the sentences in the list texts.
	- MAX_SEQUENCE_LENGTH: maximum length of the sentence to be considered, longer sentences will be terminated at this length.(default is 50)
	- MAX_NB_WORDS: maximum number of words to be used in the model (default is 10000).
	- EMBEDDING_DIM: dimension of the word vectors (default is 300 for google word2vec).
	- VALIDATION_SPLIT: fraction of data to be used for validation. (default is 0.2).

- Split into train and test data.
```
x_train, y_train, x_val, y_val, word_index = tp.preprocess()
```

- Build the embeddings index.
```
embeddings_index = tp.build_embedding_index_from_word2vec(fname, word_index)
```

- Serialize the data after the processing.
```
import cPickle

cPickle.dump([word_index, embeddings_index], open('tokenization_and_embedding.p', 'wb'))
```

- Get labels index.
```
labels_index = tp.labels_index
```

- Build the CNN model
```
from text_cnn import kimCNN

model = kimCNN(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, embeddings_index, word_index, labels_index=labels_index)
```

- Fit the model
```
model.fit(x=x_train, y=y_train, batch_size=50, epochs=25 , validation_data=(x_val, y_val))
```

For a detailed example see `example.py`. This is the same example used in Kim's paper and the original [theano code](https://github.com/yoonkim/CNN_sentence).


## References:
- [Convolution Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Theano Code](https://github.com/yoonkim/CNN_sentence)
- [Tensorflow code](https://github.com/dennybritz/cnn-text-classification-tf)