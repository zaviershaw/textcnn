# -*- coding: utf-8 -*-
# word2vec.py:生成word2vec模型

import os
import sys
import numpy as np
import gensim
import codecs
import multiprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences

max_features=200
max_document_length=200

reload(sys)
sys.setdefaultencoding( "utf-8" )
   
def load_one_file(filename):
    x=[]
    #print type(x)
    with open(filename) as f:
        for line in f:
            line=line.strip().split(' ')
           # print type(line)
            x.append(line)
        #print type(x)
    return x

class sentences_generator(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in codecs.open(self.dirname,"r", encoding="utf-8",errors="ignore"):
            yield line.strip().split()

def get_features_by_word2vec():
    # word2vec.txt数据的地址
    trainX_ham = "ham.txt"
    trainX_spam = "spam.txt"

    # 生成的word2vec模型的地址
    word2vec_bin = "word2vec.bin"
    x = []
    ham = load_one_file(trainX_ham)

   # print type(x)
    spam = load_one_file(trainX_spam)
    x = ham+spam
    y = [1]*len(ham)+[0]*len(spam)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    
    #print 1

    #cores=multiprocessing.cpu_count()
    if os.path.exists(word2vec_bin):
        print "Find cache file %s" % word2vec_bin
        model=gensim.models.Word2Vec.load(word2vec_bin)
       # print 2
    else:
        #sentences = sentences_generator(train_path)
        cores=multiprocessing.cpu_count()
    # 此处min_count=5代表5元模型，size=100代表词向量维度，worker=15表示15个线程
        model=gensim.models.Word2Vec(size=max_features, window=8, min_count=0, iter=10, workers=cores)
        model.build_vocab(x)
        #print 3

    #保存模型
    model.save(word2vec_bin)
    #print 4

    x_train=getVecsByWord2Vec(model,x_train,max_features)
    x_test = getVecsByWord2Vec(model, x_test, max_features)

    #print 5

    return x_train, x_test, y_train, y_test

def padding_sentences(sentences, padding_token, padding_sentence_length = max_document_length):
    for sentence in sentences:
        if len(sentence) > padding_sentence_length:
            sentence = sentence[:padding_sentence_length]
        else:
            sentence.extend([padding_token] * (padding_sentence_length - len(sentence)))
    return sentences

def getVecsByWord2Vec(model, corpus, size):
    global max_document_length
    print 1
    #corpus = padding_sentences(corpus,'<PADDING>',max_document_length)
    print 1
   #x=np.zeros((max_document_length,size),dtype=float, order='C')
    x=[]
    embeddingUnknown = np.zeros((1,size))
    for text in corpus:
        text_len=len(text)
        xx = []
        for i in range(0,max_document_length):
            if i<text_len:
                try:
                    xx.append (model[text[i]].reshape(1,size))
                except KeyError:
                    xx.append (embeddingUnknown)
                    continue
            else:
                xx.append(embeddingUnknown)

        x_tmp = np.concatenate(xx)
        x.append(x_tmp)

    x = np.array(x,dtype='float')
    print x.shape
    return x

def  get_features_by_tf():
    global  max_document_length
    trainX_ham = "ham.txt"
    trainX_spam = "spam.txt"

    # 生成的word2vec模型的地址
    word2vec_bin = "word2vec.bin"
    x = []
    ham = load_one_file(trainX_ham)

   # print type(x)
    spam = load_one_file(trainX_spam)
    x = ham+spam
    y = [1]*len(ham)+[0]*len(spam)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    #print x_test

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x_train=vp.fit_transform(x_train, unused_y=None)
    x_train=np.array(list(x_train))

    x_test=vp.transform(x_test)
    x_test=np.array(list(x_test))
    #print x_test
    return x_train, x_test, y_train, y_test

def do_cnn_word2vec(trainX, testX, trainY, testY):
    global max_features
    print "CNN and word2vec"

    #trainX = pad_sequences(trainX, maxlen=max_document_length, value=-1.)
    #testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_features], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=2,run_id="spam")


if __name__=="__main__":
    spam_data = "spam.txt"
    ham_data = "ham.txt"
    x_train, x_test, y_train, y_test = get_features_by_word2vec()
    #x_train,x_test,y_train,y_test = get_features_by_tf()
    #do_cnn_word2vec(x_train, x_test, y_train, y_test)
