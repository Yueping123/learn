import numpy as np
import pandas as pd
import re
import csv
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
from sklearn.ensemble import RandomForestClassifier
import time

from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


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

'''
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
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
import time
from keras import backend as K 
def focal_loss(gamma=2., alpha=0.25): 
 def focal_loss_fixed(y_true, y_pred): 
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) 
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred)) 
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)) 
 return focal_loss_fixed




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

#print(x_train)
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
print (embedding_matrix )

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                           # weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))

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
#model.compile(optimizer='rmsprop', loss=[focal_loss([6066,1987])], metrics=['accuracy'])
model.compile(loss=[focal_loss(alpha=0.5, gamma=2)],optimizer='rmsprop',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=20, batch_size=128,class_weight='auto')

#metrics.f1_score(y_true, y_pred, average='weighted') 

y_pred = model.predict(x_test, batch_size=128)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print('\n')


import pandas as pd
import time

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

label_list=[]
for review in train:
    label = review[1]
    label_list.append(label)

from numpy import *




def loadSimpleData():
    datMat = matrix(
        [[1., 2.1],
         [2., 1.1],
         [1.3, 1.],
         [1., 1.],
         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# 预测分类结果，
# dataMatrix:数据集，dimen 指的是哪个feature，threshVal：在该feature上的门限，threshIneq：
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))  # 返回的分类结果
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray  # 返回分类结果


# dataArr:数据集
# classLabels:标签集
# D:数据集合的样本的权重
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}  # 最好划分的各种信息：比如按照哪个feature划分，划分feature的门限，不等式，以及分类器权重alpha
    bestClasEst = mat(zeros((m, 1)))  # 最好划分的各个样本的预测类别
    minError = inf
    for i in range(n):  # 在数据集的所有feature上遍历
        rangeMin = dataMatrix[:, i].min()  # 找到当前feature的最小值
        rangeMax = dataMatrix[:, i].max()  # 当前feature的最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 每步骤的步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 预测分类结果到predictedVals
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 计算分类错误
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightError = D.T * errArr
                if weightError < minError:
                    minError = weightError
                    bestClasEst = predictedVals.copy()  # 最好的分类结果
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回分类错误率minError,最佳分类信息bestStump,最佳分类结果
    return bestStump, minError, bestClasEst


# adaBoosting算法
def adaBoostTrainDS(dataArr, classLabels, numIt=40):  # numIt最多有多少个弱分类器
    weakClassArr = []  # 弱分类器的集合
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 最初的数据集中的数据的权重
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 使用弱分类器分类数据集
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # 计算分类器的权重
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 增加新的弱分类器到集合中
        # 更新数据集中数据的权重
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # aggClassEst:表示最终的分类结果
        aggClassEst += alpha * classEst  # alpha表示分类器的权重
        # 分类器的错误率
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m  # errorRage是分类的错误率
        if errorRate == 0.0:  # 若分类错误率为0就reurn
            break;
    return weakClassArr


# adaBoosting测试数据
def adaClassify(datToClass, classfierArr):  # datToClass:测试的数据集合，classifierArr:弱分类器的数组
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classfierArr)):
        classEst = stumpClassify(dataMatrix, classfierArr[i]['dim'],
                                 classfierArr[i]['thresh'],
                                 classfierArr[i]['ineq'])
        aggClassEst += classfierArr[i]['alpha'] * classEst
    return sign(aggClassEst)


#dataArr, classLabels = loadSimpleData()
print (embedding_matrix )
print (label_list)
weakArr = adaBoostTrainDS(embedding_matrix , label_list)
print (weakArr)
print (adaClassify([[5, 5], [0, 0]], weakArr))

