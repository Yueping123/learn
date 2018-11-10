from __future__ import absolute_import
from __future__ import print_function
from nltk.tokenize import word_tokenize

import nltk
import csv
import re
import sys
import matplotlib.pyplot as plt
import os
import logging
import gensim
import pickle
import numpy as np
import pandas as pd
import nltk.tokenize as tk
import seaborn as sns
import matplotlib



from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict


train ="C:\\Users\\YP\\Desktop\\pro\\Training_A.csv"
test= "C:\\Users\\YP\\Desktop\\pro\\test_A.csv"

def read_csv(data_path):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    sent_list = []
    for row in file_reader:
        sent = row[1]
        score = row[2]
        sent_list.append((sent,score))
    return sent_list
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
    return (words5)


def build_data_train_test(data_train, data_test, train_ratio=0.8):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set
    sence=0
    for i in range(len(data_train)):
        #print(i)
        sence=sence+1
        rev = data_train[i]
        y = data_train[i][-1]
        #print(y)
        orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'type': 'traindata',
                 'sence':sence,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        print(datum)
        revs.append(datum)
    print(len(data_test))
    for i in range(len(data_test)):
        #print (i)
        rev = data_test[i]
        #(rev)
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                     'type':'testdata',
                     'text': orig_rev,
                     'num_words': len(orig_rev.split()),
                     'split': -1}
        print(datum)
        revs.append(datum)

    return revs, vocab


def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    logging.info('unk words: %d' % (unk_words))
    return word_vecs


def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map

if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])  # 用到os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值。
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    sent_list = read_csv(train)
    test=read_csv(test)
    sent_list1 = []
    for review in test:
        sent = review[0]
        sent_list1.append(sent)
    print (sent_list1)

    clean_train_reviews = []
    for review in sent_list:
        clean_train_reviews.append(clean_data(review))
    clean_test_reviews = []
    for review in test:
        clean_test_reviews.append(clean_data(review))


    revs, vocab = build_data_train_test(clean_train_reviews, clean_test_reviews)
    max_l= np.max(pd.DataFrame(revs)['num_words'])
    print (len(revs))
    print(max_l)

    # 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
    t_sence=[]
    t_words=[]
    for rev in revs:
        if (rev['type'] == 'traindata'):
            t_sence.append(rev['sence'])
            t_words.append(rev['num_words'])
            print(rev['sence'],rev['num_words'])
    plt.bar(t_sence,t_words)
    plt.show()


