# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:16:19 2017

@author: 颜

"""
import numpy as np
import os
import collections
import random

def readfile(file):
    if not os.path.exists(file):
        print("File Not Exist!!!")
        os._exit()
    else:
        f=open(file)
    word_list=f.read().split()
    print("File Loaded!!!\nLength=%d"%(len(word_list)))
    return word_list

def init_dictionary(word_list,dictionary_size=0):
    dictionary=dict()
    count=[['UNK',0]]
    count.extend(collections.Counter(word_list).most_common(dictionary_size-1))
    index=0
    for word in count:
        dictionary[word[0]]=index
        index+=1
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    word_number=[]
    lowfreq_word_num=0
    for word in word_list:
        if word in dictionary:
            word_number.append(dictionary[word])
        else:
            word_number.append(0)
            lowfreq_word_num+=1
        count[0][1]=lowfreq_word_num
    return dictionary,reverse_dictionary,count,word_number

def negative_sample(sample_num,sample_from,probability):
    word=np.arange(0,len(sample_from),1)
    sample=np.random.choice(word,sample_num,p=probability)
    return sample

def sample_probability(sample_from):
    word_freq=np.array([float(x[1]) for x in sample_from])
    regulize=np.add.reduce(word_freq)
    probabilities=word_freq/regulize
    probabilities[0]+=(1-np.add.reduce(probabilities,0))
    return probabilities

def generate_batch(word_number,count,batch_size,pin,context_window=1,sample_num=1,sample_window=100,sample_probabilities=None):
    assert context_window<=sample_num*2
    assert batch_size%2==0
    length=len(word_number)
    pin=pin%length
    batch=np.array([-1]*batch_size)
    labels=np.array([-1]*batch_size*(sample_num)).reshape([batch_size,sample_num])
    labels_mark=np.array([0.]*batch_size*(sample_num)).reshape([batch_size,sample_num])
    window=context_window*2+1
    deque=collections.deque(maxlen=window)
    for i in range(window):
        deque.append(word_number[pin])
        pin=(pin+1)%length
    for i in range(batch_size):
        batch[i]=deque[context_window]
        positive_label_num=0
        while positive_label_num==0:
            positive_label_num=random.randint(-context_window,context_window)
        mark=abs(positive_label_num)//positive_label_num
        positive_label_num=abs(positive_label_num)
        for j in range(positive_label_num):
            labels[i][j]=deque[context_window+mark*(1+j)]
            labels_mark[i][j]=1.
        sample=negative_sample(sample_num-positive_label_num,count[0:sample_window],sample_probabilities)
        labels[i,positive_label_num:]=sample
        deque.append(word_number[pin])
        pin=(pin+1)%length
    return batch,labels,labels_mark,pin








