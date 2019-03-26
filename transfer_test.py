# -*- coding: utf-8 -*-
#transfer.py:生成spam和ham数据
import jieba
import sys
import os
import re

# 判断邮件中的字符是否是中文
def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

# 加载邮件数据的label
def load_label_files(label_file):
    label_dict ={}
    for line in open(label_file).readlines():
        list1 = line.strip().split("../")
        label_dict[list1[1].strip()] = list1[0].strip()
    return label_dict

# 加载停用词词表
def load_stop_train(stop_word_path):
    stop_dict = {}
    for line in open(stop_word_path).readlines():
        line = line.strip()
        stop_dict[line] = 1
    return stop_dict

# 读取邮件数据，并转换为utf-8格式，生成spam和ham样本
def read_files(file_path,label_dict,stop_dict,spam_file_path,ham_file_path):
    parents = os.listdir(file_path)
    spam_file = open(spam_file_path,'a')
    ham_file = open(ham_file_path,'a')
    for parent in parents:
        child = os.path.join(file_path,parent)
        if os.path.isdir(child):
            read_files(child,label_dict,stop_dict,spam_file_path,ham_file_path)
        else:
            print child[7:]
            label = "unk"
            if child[7:] in label_dict:
                label = label_dict[child[7:]]
            # deal file
            temp_list = []
            for line in open(child).readlines():
                line = line.strip().decode("gbk",'ignore').encode('utf-8')
                if not check_contain_chinese(line):
                    continue
                seg_list = jieba.cut(line, cut_all=False)
                for word in seg_list:
                    if word in stop_dict:
                        continue
                    else:
                        temp_list.append(word)
            line = " ".join(temp_list)
            print label
            if label == "spam":
                spam_file.write(line.encode("utf-8","ignore") + "\n")
            if label == "ham":
                ham_file.write(line.encode("utf-8","ignore")+"\n")

# 生成word2vec词表
def generate_word2vec(file_path,label_dict,stop_dict,word_vec):
    parents = os.listdir(file_path)
    fh1 = open(word_vec,'a')
    i = 0

    for parent in parents:
        child = os.path.join(file_path,parent)
        if os.path.isdir(child):
            generate_word2vec(child,label_dict,stop_dict,word_vec)
        else:
            print child[7:]
            label = "unk"
            if child[7:] in label_dict:
                label = label_dict[child[7:]]
            # deal file
            temp_list = []
            for line in open(child).readlines():
                line = line.strip().decode("gbk",'ignore').encode('utf-8')
                if not check_contain_chinese(line):
                    continue
                if len(line) == 0:
                    continue
                seg_list = jieba.cut(line, cut_all=False)
                for word in seg_list:
                    if word in stop_dict:
                        continue
                    else:
                        temp_list.append(word)
            line = " ".join(temp_list)
            fh1.write(line.encode("utf-8","ingore")+"\n")

if __name__=="__main__":
    file_path = sys.argv[1]
    label_path = sys.argv[2]
    stop_word_path = "stop_words.txt"
    word_vec_path = "word2vec.txt"
    spam_data = "spam.txt"
    ham_data = "ham.txt"
    label_dict = load_label_files(label_path)
    stop_dict = load_stop_train(stop_word_path)
    read_files(file_path,label_dict,stop_dict,spam_data,ham_data)
    generate_word2vec(file_path,label_dict,stop_dict,word_vec_path)
