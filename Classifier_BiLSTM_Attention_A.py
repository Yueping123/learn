import numpy as np
import pandas as pd
import re
import csv
import os
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,Dropout,GRU
from keras.layers import  Embedding, LSTM, Bidirectional
from keras.models import  Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import codecs
class Attention_layer(Layer):

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print (a)
        # a = K.expand_dims(a)
        print (x)
        weighted_input = x * a
        print (weighted_input)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



MAX_SEQUENCE_LENGTH = 80
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


def focal_loss(classes_num, gamma=2.0, alpha=0.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        
       # prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
       # target_tensor is the label tensor, same shape as predcition_tensor
        
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(target_tensor > zeros, classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed
'''

from keras import backend as K  
def focal_loss(gamma=2.0, alpha=0.25): 
 def focal_loss_fixed(y_true, y_pred): 
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) 
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred)) 
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)) 
 return focal_loss_fixed 
'''


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
 
 def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:' ,_val_f1)
        return _val_f1

metrics = Metrics()
#print (metrics.val_f1s)

from keras import backend as K
 
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from datetime import datetime
from time import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def train(metrics, model, batch, epoch, data_augmentation=False):
    start = time()
    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    
    es = EarlyStopping(monitor='_val_f1', patience=20)
    mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_f1:.4f}.h5', 
                         monitor='_val_f1', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)
    
    if data_augmentation:
        aug = ImageDataGenerator(width_shift_range = 0.125, height_shift_range = 0.125, horizontal_flip = True)
        aug.fit(x_train)
        gen = aug.flow(x_train, y_train, batch_size=batch)
        h = model.fit_generator(generator=gen, 
                                steps_per_epoch=50000/batch,
                                epochs=epoch, 
                                validation_data=(x_val, y_val),
                                callbacks=[es, mc, tb])
    else:
        start = time()
        h = model.fit(#x=x_train, 
                      #y=y_train, 
                      batch_size=None,  
                      epochs=epoch,
                      steps_per_epoch=12027/batch,
                      validation_steps=50,
                      validation_data=(x_val,y_val),
                      callbacks=[metrics,es, mc, tb])
    
    print('\n@ Total Time Spent: %.2f seconds' % (time() - start))
    acc, val_acc = h.history['acc'], h.history['val_acc']
    _val_f1=h.history['_val_f1']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    return h





def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def clean_data(sentence1):
    words = str(sentence1).strip().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    words1 = " ".join(meaningful_words)
    words2 = words1.lower().strip().split()
    clean_sentence = []
    for word in words2:
        # print(word)
        if 'http://' in word:
            word = 'url'
        elif 'https://' in word:
            word = 'url'
        elif '@' in word:
            word = ' '
        else:
            word = word
        clean_sentence.append(word)
    review_text = re.sub("[^a-zA-Z0-9]", " ", str(clean_sentence))  # 去掉除英文字母的其他字符
    words5 = review_text.lower().strip().split()
    words6 = " ".join(words5)

    return (words6)

def read_csv(data_path):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    data_train = []
    for row in file_reader:
        sent = row[1]
        score = row[2]
        data_train.append((sent,score))
    return data_train
data_train=read_csv("Training_A.csv")
test=read_csv("Test_A.csv")

#data_train = pd.read_csv('label_data.tsv', sep='\t')
print (len(data_train))
texts = []
labels = []

for i in range(len(data_train)):
    text = BeautifulSoup(data_train[i][0], "lxml")
    text1=clean_str(text.get_text())
    texts.append(clean_data(text1))
    labels.append(data_train[i][1])
#print (texts,labels)

x_tests = []
for i in range(len(test)):
    text= BeautifulSoup(test[i][0], "lxml")
    x_tests.append(clean_str(text.get_text()))
#print (x_test)

embeddings_index = {}
f = open('glove.6B.300d.txt',encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype="float32")
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

labels = to_categorical(np.asarray(labels),num_classes=2)
print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))


tokenizer = Tokenizer(nb_words=None)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
tokenizer.fit_on_texts(x_tests)
sequences1 = tokenizer.texts_to_sequences(x_tests)


word_index = tokenizer.word_index


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
test1=pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)


indices = np.arange(data.shape[0])  # make a list ,lenght is 25000
indices1=np.arange(test1.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_test=test1[indices1]

#print(test1)
print ('\n')


# SPLIT DATA , ONE PART FOR TRAIN , THE OTHER PART FOR PREDICT
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

#print(y_train)
print('\n')
print (x_train.shape)
print (y_train.shape)
print (x_val.shape)
print (y_val.shape)


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print ('Length of embedding_matrix:', embedding_matrix.shape[0])

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                           # weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))
'''
#LSTM+attention
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
l_att = Attention_layer()(l_gru)
dense_1 = Dense(100,activation='tanh')(l_att)
print ("************************************************")
print (dense_1.shape)
print ("************************************************")
dense_2 = Dense(2, activation='softmax')(dense_1)
print ("************************************************")
print (dense_2.shape)
model = Model(sequence_input, dense_2)
print ("************************************************")
#model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=[f1])
#model.compile(optimizer='rmsprop', loss=[focal_loss([6066,1987])], metrics=['accuracy'])
#model.compile(loss=focal_loss(alpha=0.5, gamma=2.0),optimizer='rmsprop',metrics=['accuracy'])
#model.summary()
#model.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=20, batch_size=128,class_weight='auto')
#model.fit(x_train,y_train, validation_data=(x_val, y_val),nb_epoch=10,batch_size=128,callbacks=[metrics],class_weight={0:0.25,1:0.75})
#model.fit(x_train,y_train, validation_data=(x_val, y_val),nb_epoch=10,batch_size=128,callbacks=[metrics])

# early stoppping
from keras.callbacks import EarlyStopping
start = time()
log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
os.mkdir(log_dir)
es = EarlyStopping(monitor='f1', patience=20)
mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{f1:.4f}.h5',
                         monitor='f1', save_best_only=True)
tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

early_stopping = EarlyStopping(monitor='f1', patience=50, verbose=2)
# 训练
history = model.fit(x_train,y_train, epochs=50, batch_size=20, validation_data=(x_val,y_val), verbose=2, shuffle=False, callbacks=[early_stopping,metrics,mc,tb])

#h=train(metrics, model,93,epoch)

y_pred = model.predict(x_test, batch_size=128)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print('\n')
'''



from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10
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



batch_size = 100
nb_epoch = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 60


print("*****************cnn_model")
 # Keras Model for cnn
 # this is the placeholder tensor for the input sequence
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
def cnn(sequence_input):
    embedded = embedding_layer(sequence_input)
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
    model = Model(inputs=sequence_input, outputs=output)
    return model

model=cnn(sequence_input)


start = time()
log_dir_cnn = datetime.now().strftime('cnn_model_%Y%m%d_%H%M')
os.mkdir(log_dir_cnn)
filepath_cnn= log_dir_cnn + '\\CIFAR10-cnn.h5'


def cnn_compile_and_train(filepath,log_dir,model, num_epochs):
 
   model.compile(optimizer='rmsprop', loss=[focal_loss([6066,1987])], metrics=[f1])

 #  model.compile(loss=focal_loss(alpha=0.5, gamma=2.0), optimizer=('Adam'), metrics=[f1])
  # start = time()
  # log_dir = datetime.now().strftime('cnn_model_%Y%m%d_%H%M')
  # os.mkdir(log_dir)
  # filepath= log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{f1:.4f}.h5'
   es = EarlyStopping(monitor='f1', patience=20)
   mc = ModelCheckpoint(filepath,monitor='f1', save_best_only=True)
   tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

   early_stopping = EarlyStopping(monitor='f1', patience=50, verbose=2)
   
   history=model.fit(x_train,y_train, validation_data=[x_val, y_val], batch_size=batch_size, epochs=num_epochs, verbose=2,callbacks=[metrics,early_stopping,mc,tb])

   return history

cnn_compile_and_train(filepath_cnn,log_dir_cnn,model, num_epochs=100)

def evaluate_cnn_error(model):

   pred =model.predict(x_test,batch_size =batch_size)

   pred =np.argmax(pred,axis=1)

   #pred =np.expand_dims(pred,axis=1)# make same shape as y_test

   #error =np.sum(np.not_equal(pred,y_test))/y_test.shape[0]

   return pred

y_pred_cnn=evaluate_cnn_error(model)

#print(y_pred_cnn)


print("******************************lstm_model")

#keras model for lstm+attention
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
def LSTM_Attention(sequence_input):
   embedded_sequences = embedding_layer(sequence_input)
   l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
   l_att = Attention_layer()(l_gru)
   dense_1 = Dense(100,activation='sigmoid')(l_att)
   print ("************************************************")
   print (dense_1.shape)
   dense_2 = Dense(2, activation='softmax')(dense_1)
   print ("************************************************")
   print (dense_2.shape)
   model = Model(sequence_input, dense_2)
   return model

model=LSTM_Attention(sequence_input)

start = time()
log_dir_lstm = datetime.now().strftime('lstm_model_%Y%m%d_%H%M')
os.mkdir(log_dir_lstm)
filepath_lstm=log_dir_lstm + '\\CIFAR10_lstm.h5'

def lstm_compile_and_train(filepath,log_dir,model, num_epochs):
 # model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=[f1])
  # model.compile(loss=focal_loss(alpha=0.5, gamma=2.0),optimizer='rmsprop',metrics=[f1])
   model.compile(optimizer='rmsprop', loss=[focal_loss([6066,1987])], metrics=[f1])

 # start = time()
 # log_dir = datetime.now().strftime('lstm_model_%Y%m%d_%H%M')
 # os.mkdir(log_dir)
 # filepath1=log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{f1:.4f}.h5'

   es = EarlyStopping(monitor='f1', patience=20)
   mc = ModelCheckpoint(filepath, monitor='f1', save_best_only=True)
   tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

   early_stopping = EarlyStopping(monitor='f1', patience=50, verbose=2)
# 训练
   history = model.fit(x_train,y_train, epochs=num_epochs, batch_size=20, validation_data=(x_val,y_val), verbose=2, shuffle=False, callbacks=[early_stopping,metrics,mc,tb])

   return history

lstm_compile_and_train(filepath_lstm,log_dir_lstm,model, num_epochs=100)

def evaluate_lstm_error(model):

   pred =model.predict(x_test,batch_size =batch_size)

   pred =np.argmax(pred,axis=1)

  # pred =np.expand_dims(pred,axis=1)# make same shape as y_test

   #error =np.sum(np.not_equal(pred,y_test))/y_test.shape[0]

   return pred

y_pred_lstm=evaluate_lstm_error(model)
#print (y_pred_lstm)

#集成模型
cnn_model =cnn(sequence_input)

lstm_model =LSTM_Attention(sequence_input)

cnn_model.load_weights(filepath_cnn)

lstm_model.load_weights(filepath_lstm)

models =[cnn_model,lstm_model]

#集成模型的定义是很直接的。它使用了所有模型共享的输入层。在顶部的层中，该集成通过使用 Average() 合并层计算三个模型输出的平均值。

def ensemble(models,sequence_input):

  outputs =[model.outputs[0] for model in models]

  y =Average()(outputs)

  model =Model(sequence_input,y,name='ensemble')

  return model

ensemble_model =ensemble(models,sequence_input)


def evaluate_error(model):

   pred =model.predict(x_test,batch_size =batch_size)

   pred =np.argmax(pred,axis=1)

  # pred =np.expand_dims(pred,axis=1)# make same shape as y_test

   #error =np.sum(np.not_equal(pred,y_test))/y_test.shape[0]

   return pred

y_pred=evaluate_error(ensemble_model)
#print (y_pred)

label_list=[]
for y in y_pred:
    # print (y)
    label_list.append(y)
print (label_list)


#将结果写入csv文件
words1=[]
i=0
for review in test:
        id = review[0]
        # print(id)
        sentence2 = review[1]
        words1.append((id, sentence2, label_list[i]))
        file_csv = codecs.open("sub_A.csv", 'w+', 'utf-8')  # 追加
        writer = csv.writer(file_csv, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for data in words1:
            writer.writerow(data)
        i=i+1
