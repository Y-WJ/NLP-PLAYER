# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:05:46 2017

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
    count=[['lowfreq_word',-1]]
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

def generate_list(matrix_size,word_number,context_window):
    length=len(word_number)
    print("Statistic Begin!!!")
    length=len(word_number)
    i=0
    X=np.zeros([matrix_size,matrix_size],dtype=int)
    list_X=[]
    while i<length-1-context_window:
        j=i
        for k in range(context_window):
            j+=1
            if X[word_number[i]][word_number[j]]==0:
                list_X.append([word_number[i],word_number[j],0])
                list_X.append([word_number[j],word_number[i],0])
            X[word_number[i],word_number[j]]+=1
            X[word_number[j],word_number[i]]+=1
        i+=1
        if i%10000==0:
            print("process",i*100/length,"%")
    print("Statistic Finished!!!")
    print("Generating List!!!")
    for term in list_X:
        term[2]=X[term[0],term[1]]

    print("lenght=",len(list_X))
    return np.array(list_X)

def generate_batch(list_X,batch_size):
    choice=np.random.choice(len(list_X),batch_size)
    batch=list_X[choice,:]
#    if pin+batch_size>length:
#        pin=0
#    batch=list_X[pin:pin+batch_size,:]
#    pin+=batch_size
#    else:
#        batch=np.row_stack((list_X[pin:length,:],list_X[0:batch_size+pin-length,:]))
#        pin=batch_size+pin-length
    word_i=batch[:,0]
    word_j=batch[:,1]
    X_ij=batch[:,2]
    return word_i,word_j,X_ij






