# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import csv

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adadelta
# from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence
from keras.layers import Input, merge
from keras.optimizers import RMSprop, Adagrad, Adadelta

from keras.utils import np_utils


# maxlen = 56
batch_size = 100
nb_epoch = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 60

test1= "C:\\Users\\YP\\Desktop\\pro\\test_A.csv"
out_path2 ="C:\\Users\\YP\\Desktop\\pro\\"+"test_prediction.csv"
def read_csv(data_path):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    sent_list = []
    for row in file_reader:
        sent = row[1]
        score = row[2]
        sent_list.append((sent,score))
    return sent_list

def read_csv_id(data_path):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    sent_list= []
    for row in file_reader:
        id= row[0]
        sent_list.append((id))
    return sent_list

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x

def make_idx_data(revs, word_idx_map, maxlen=80):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev]

def write_csv(ids,sent_list, label_list, out_path):
        filewriter = csv.writer(open(out_path, "w+"))
        count = 0
        for (id,sent, label) in (ids,sent_list, label_list):
            filewriter.writerow([id, sent, label])
 
if __name__ == '__main__':
    test = read_csv(test1)
    id = read_csv_id(test1)
    print(id)
    print(len(id))
    sent_list1 = []
    for review in test:
        sent = review[0]
        sent_list1.append(sent)
    #print(sent_list1)
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    # pickle_file = sys.argv[1]
    pickle_file = os.path.join('pickle', 'imdb_train_test_A.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_test_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = Input(shape=(maxlen, ), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)

    # convolutional layer
    convolution = Convolution1D(filters=nb_filter,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1
                            ) (embedded)

    maxpooling = MaxPooling1D(pool_size=2) (convolution)
    maxpooling = Flatten() (maxpooling)

    # We add a vanilla hidden layer:
    dense = Dense(70) (maxpooling)    # best: 120
    dense = Dropout(0.25) (dense)    # best: 0.25
    dense = Activation('relu') (dense)

    output = Dense(2, activation='softmax') (dense)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2)
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    model.summary()
    model.get_weights()
    print('\n')
    print(y_pred)
    #result_output = pd.DataFrame(y_pred,sent_list1)
    #print (result_output)
    # Use pandas to write the comma-separated output file
    #result_output.to_csv("submission_A.csv")
    write_csv(id,sent_list1,y_pred, out_path2)
