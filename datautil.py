from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from collections import Counter
import numpy as np
import random

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def build_vocab(data_path,vocab_path):

    if os.path.exists(vocab_path):
        print(" Using existed vocab !")
        return
    vocab = {}
    f = open(data_path, 'r', encoding='utf-8')
    fw = open(vocab_path, 'w', encoding='utf-8')
    for line in f.readlines():
        if line.strip():
            line_pair = line.split('|')
            line_query = line_pair[0]
            line_answer = line_pair[1]
            for word in line_query.split():
                if word not in vocab.keys():
                    vocab[word] = 1
            for word in line_answer.split():
                if word not in vocab.keys():
                    vocab[word] = 1
    for word in vocab.keys():
        fw.write(word + '\n')
    f.close()
    fw.close()


def load_vocab(vocab_path):
    f = open(vocab_path, 'r', encoding='utf-8')
    rev_vocab = [_PAD,_GO,_EOS,_UNK]
    rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict(zip(rev_vocab,range(len(rev_vocab))))

    vocab_size = len(rev_vocab)

    return vocab, rev_vocab,vocab_size

def sentence_to_ids(sentence,vocab,eos_flag=False):
    words = sentence.strip().split()
    if eos_flag:
        words.append(_EOS)
    return [vocab.get(w,UNK_ID) for w in words]

def map_file_to_ids(filename,filename_ids,vocab):
    if not os.path.exists(filename_ids):
        with open(filename,'r',encoding='utf-8')as f,open(filename_ids,'w')as fw:
            for line in f:
                items = line.split('|')
                if len(items)<2:
                    continue
                ids0 = sentence_to_ids(items[0],vocab,eos_flag=False)
                ids1 = sentence_to_ids(items[1],vocab,eos_flag=False)
                if len(ids0)>=1 and len(ids1)>=1:
                    fw.write('%s|%s\n'%(' '.join(map(str,ids0)),' '.join(map(str,ids1))))
                    fw.flush()

    else:
        print("Using exsited id file!")

def prepare_data(data_dir,vocab_path):
    train_path = os.path.join(data_dir,'train.data')
    valid_path = os.path.join(data_dir,'valid.data')

    build_vocab(train_path,vocab_path)
    vocab,rev_vocab,vocab_size = load_vocab(vocab_path)

    train_ids_path = train_path+".ids.txt"
    valid_ids_path = valid_path + ".ids.txt"

    map_file_to_ids(train_path,train_ids_path,vocab)
    map_file_to_ids(valid_path,valid_ids_path,vocab)

    return train_ids_path,valid_ids_path,vocab_path,vocab_size

def prepare_test_data(data_dir,vocab_path):
    test_path = os.path.join(data_dir,'test.data')

    vocab,rev_vocab,vocab_size = load_vocab(vocab_path)
    test_ids_path = test_path+".ids.txt"
    map_file_to_ids(test_path,test_ids_path,vocab)

    return test_ids_path,vocab,rev_vocab,vocab_size

def padding(samples,max_len=20,dtype = 'int32'):
    sample_num = len(samples)
    x = (np.ones((sample_num,max_len))*0.).astype(dtype)
    len_xs = []
    y = (np.ones((sample_num,max_len))*0.).astype(dtype)
    y_weight = (np.ones((sample_num,max_len))*1.).astype('float32')
    for idx,(s,sy) in enumerate(samples):
        len_x = len(s)
        x[idx,:len_x] =s
        len_y = len(sy)
        y[idx,:len_y] = sy
        y_weight [idx,len_y:] = 0
        len_xs.append(len_x)

    return x,len_xs,y,y_weight

class Itertool(object):

    def __init__(self,data_path,batch_size=128,max_len=20,eos_flag=True,go_flag=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.eos_flag = True
        self.go_flag = True

    def __iter__(self):
        with open(self.data_path,'r')as f:
            sample_num = 0
            samples = []
            for line in f:
                sample_num += 1
                items = line.split('|')
                query = items[0].strip()
                answer = items[1].strip()
                query = query.split()
                answer = answer.split()

                if self.eos_flag:
                    query.append(EOS_ID)
                    answer.append(EOS_ID)
                if len(query) > self.max_len:
                    query = query[:self.max_len]
                if len(answer) > self.max_len:
                    answer = answer[:self.max_len]
                samples.append((query,answer))
                if sample_num == self.batch_size:
                    x,len_x,y,y_weight = padding(samples,max_len=self.max_len)
                    yield x,len_x,y,y_weight
                    sample_num = 0
                    samples = []
            if len(samples)>0:
                x,len_x,y,y_weight = padding(samples,max_len=self.max_len)
                yield x,len_x,y,y_weight











