# -*- coding: utf-8 -*-
# word2vec.py:生成word2vec模型

import os
import sys
import numpy as np
import gensim
import codecs
import multiprocessing

max_features=200

reload(sys)
sys.setdefaultencoding( "utf-8" )

class sentences_generator(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in codecs.open(self.dirname,"r", encoding="utf-8",errors="ignore"):
            yield line.strip().split()

def get_features_by_word2vec():
    # word2vec.txt数据的地址
    train_path = "word2vec.txt"

    # 生成的word2vec模型的地址
    word2vec_bin = "word2vec.bin"

    #sentences = sentences_generator(train_path) 
    #cores=multiprocessing.cpu_count()
    if os.path.exists(word2vec_bin):
        print "Find cache file %s" % word2vec_bin
        model=gensim.models.Word2Vec.load(word2vec_bin)
    else:
        sentences = sentences_generator(train_path)
        cores=multiprocessing.cpu_count()
    # 此处min_count=5代表5元模型，size=100代表词向量维度，worker=15表示15个线程
        model=gensim.models.Word2Vec(size=max_features, window=8, min_count=5, iter=10, workers=cores)
        model.build_vocab(sentences)

    #保存模型
    model.save(word2vec_bin)

def getVecsByWord2Vec(model, corpus, size):
    global max_document_length
    #x=np.zeros((max_document_length,size),dtype=float, order='C')
    x=[]

    for text in corpus:
        xx = []
        for i, vv in enumerate(text):
            try:
                xx.append(model[vv].reshape((1,size)))
            except KeyError:
                continue

        x = np.concatenate(xx)

    x=np.array(x, dtype='float')
    return x

if __name__=="__main__":
    spam_data = "spam.txt"
    ham_data = "ham.txt"
    get_features_by_word2vec()
