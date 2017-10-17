# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:15:28 2017

@author: 颜
"""

import os
from copy import copy
import tqdm
import wordtovector as wv

class CONLL():
    def __init__(self,path):
        if not(os.path.isfile(path) and os.path.splitext(path)[1]==".conllu"):
            print("Invalid File Name!!!")
            os._exit
        else:
            file=open(path  ,'rb')
            file_split=file.read().split(b'# ')
        print("INIT DATA!!!")
        self.length=0
        self.sentence=list()
        self.parsing=list()
        self.transaction=list()
        self.tran_dic=dict()
        self.sample_list=list()
        self.sample_pin=0
        self.sample_period=0
        for i in tqdm.tqdm(range(len(file_split)),desc="Loading",ncols=105,ascii=True,unit="sentence"):
            block_split=file_split[i].split(b'\n')
            p=0
            parsing_item=list()
            sentence_item=list()
            for line in block_split:
                block_line_split=line.split(b'\t')
                index=block_line_split[0]
                if index==str(p+1).encode():
                    p+=1
                    block_line_split[7]=block_line_split[7].split(b':')[0]
                    parsing_item.append(block_line_split)
                    sentence_item.append(block_line_split[2])
            if len(sentence_item)>0:
                self.sentence.append(sentence_item)
            if len(parsing_item)>0:
                self.parsing.append(parsing_item)
                a,b=self.get_transaction(parsing_item)
                self.transaction.append(a)
                self.sample_list.extend(b)
        self.length=len(self.parsing)
        v=0
        for item in self.transaction:
            for tran in item:
                if len(tran)==2:
                    t=tran[0]+tran[1]
                else:
                    t=tran
                if t not in self.tran_dic.keys():
                    self.tran_dic[t]=v
                    v+=1

    def get_text(self):
        text_orin=list()
        text_norm=list()
        self.text=list()
        for i in self.parsing:
            for j in i:
                text_orin.append(j[1])
                text_norm.append(j[2])
        self.text.extend(text_orin)
        self.text.extend(text_norm)
        return self.text

    def get_transaction(self,item):
        stack=list()
        transaction=list()
        sample_list=list()
        k=0
        def isdependent(n,k,item):
            b=True
            for i in range(k,len(item)):
                if item[i][6]==item[int(n)-1][0]:
                    b=False
            return b
        for i in range(len(item)*2):
            p=len(stack)
            stack_word=list()
            buffer_word=list()
            tran_list=list()
            for j in stack:
                stack_word.append(item[int(j[0])-1][1])
            for j in range(k,len(item)):
                buffer_word.append(item[j][1])
            if p<2:
                if len(item)==k and p==1:
                    transaction.append(['r',stack[0][2].decode()])
                else:
                    stack.append([item[k][0],item[k][6],item[k][7]])
                    k+=1
                    transaction.append('shift')
                    tran_list=copy(transaction)
            else:
                if stack[p-1][1]==stack[p-2][0] and isdependent(stack[p-1][0],k,item):
                    transaction.append(['r',stack[p-1][2].decode()])
                    tran_list=copy(transaction)
                    stack.remove(stack[p-1])
                elif stack[p-1][0]==stack[p-2][1] and isdependent(stack[p-2][0],k,item):
                    transaction.append(['l',stack[p-2][2].decode()])
                    tran_list=copy(transaction)
                    stack.remove(stack[p-2])
                elif len(item)>k:
                    stack.append([item[k][0],item[k][6],item[k][7]])
                    k+=1
                    transaction.append('shift')
                    tran_list=copy(transaction)
            sample_list.append([stack_word,buffer_word,tran_list])
        return transaction,sample_list

    def get_parsing_item(transaction,sentence):
        l=len(transaction)//2
        import numpy as np
        item=np.full([l,10],b'_').tolist()
        for k in range(l):
            item[k][0]=str(k+1).encode()
            item[k][1]=sentence[k]
            item[k][2]=sentence[k]
        stack=list(['0','0','0'])
        i=0
        for tran in transaction:
            if tran=='shift':
                stack.append([item[i][0],item[i][6],item[i][7]])
                i+=1
            elif len(tran)==2:
                p=len(stack)
                if tran[0]=='r':
                    item[int(stack[p-1][0])-1][6]=stack[p-2][0]
                    item[int(stack[p-1][0])-1][7]=tran[1].encode()
                    stack.remove(stack[p-1])
                elif tran[0]=='l':
                    item[int(stack[p-2][0])-1][6]=stack[p-1][0]
                    item[int(stack[p-2][0])-1][7]=tran[1].encode()
                    stack.remove(stack[p-2])
        return item

    def write_to_conll(self,path,parsing_item):
        if not os.path.splitext(path)[1]==".conllu":
            print("Invalid File Name!!!")
            os._exit
        content=b"# text = "
        for line in parsing_item:
            content+=line[1]
            content+=b' '
        content+=b'\n'
        for line in parsing_item:
            li=line[0]
            for p in range(9):
                li+=b'\t'
                li+=line[p+1]
            li+=b'\n'
            content+=li
        content+=b'\n'
        file=open(path,"ab+")
        file.write(content)
        file.close()

    def save(self,path):
        if not os.path.splitext(path)[1]==".conllu":
            print("Invalid File Name!!!")
            os._exit
        file=open(path,"wb+")
        for parsing_item in self.parsing:
            content=b"# text = "
            for line in parsing_item:
                content+=line[1]
                content+=b' '
            content+=b'\n'
            for line in parsing_item:
                li=line[0]
                for p in range(9):
                    li+=b'\t'
                    li+=line[p+1]
                li+=b'\n'
                content+=li
            content+=b'\n'
            file.write(content)
        file.close()

    def generate_batch(self,batch_size):
        l=self.length
        pin=self.sample_pin
        batch_sentence=[]
        batch_sentence_rev=[]
        batch_tran=[]
        length=[]
        for i in range(batch_size):
            if pin>=l:
                pin=pin-l
                self.sample_period+=1
            sentence=self.sentence[pin]
            tran=self.transaction[pin]
            pin+=1







            sentence_embedding=wv.embed(sentence)
            sentence_embedding_rev=sentence_embedding[:]
            sentence_embedding_rev.reverse()
            #tran_embedding=self.embed(tran)
            batch_sentence.append(sentence_embedding)
            batch_sentence_rev.append(sentence_embedding_rev)
            #batch_tran.append(tran_embedding)
            length.append(len(sentence))
        length.sort(reverse=True)
        import torch
        from torch.nn.utils.rnn import pack_padded_sequence
        from torch.autograd import variable
        r1=variable(torch.Tensor(batch_sentence))
        r2=variable(torch.Tensor(batch_sentence_rev))
        r3=variable(torch.Tensor(batch_tran))
        r1=pack_padded_sequence(r1,length,batch_first=True)
        r2=pack_padded_sequence(r2,length,batch_first=True)
        r3=pack_padded_sequence(r3,length,batch_first=True)
        return r1,r2,r3


