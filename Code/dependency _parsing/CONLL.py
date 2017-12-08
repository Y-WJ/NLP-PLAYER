# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:15:28 2017

@author: 颜
"""

import os
from copy import copy
import tqdm
import numpy as np

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
        self.tran_dic_rev=dict()
        self.pos_dic=dict()
        self.tran_size=0
        self.sample_list=list()
        self.sample_pin=0
        self.sample_period=0
        self.total_sentence=0
        self.sentence_index=0
        for i in tqdm.tqdm(range(len(file_split)),desc="Loading",ncols=105,ascii=True,unit="sentence"):
            block_split=file_split[i].split(b'\n')
            p=0
            parsing_item=list()
            sentence_item=list()
            v=0
            for line in block_split:
                block_line_split=line.split(b'\t')
                index=block_line_split[0]
                if index==str(p+1).encode():
                    p+=1
                    block_line_split[7]=block_line_split[7].split(b':')[0]
                    pos=block_line_split[4]
                    if pos not in self.pos_dic:
                        self.pos_dic[pos]=v
                        v+=1
                    if block_line_split[2]==b'","':
                        block_line_split[2]=','
                    parsing_item.append(block_line_split)
                    sentence_item.append(block_line_split[2])
            if len(sentence_item)>0:
                self.sentence.append(sentence_item)
                self.parsing.append(parsing_item)
                a,b=self.get_transaction(parsing_item,self.total_sentence)
                self.transaction.append(a)
                self.sample_list.extend(b)
                self.total_sentence+=1
        self.length=len(self.sample_list)
        v=0
        for item in self.transaction:
            for tran in item:
                if len(tran)==2:
                    t=tran[0]+tran[1]
                else:
                    t=tran
                if t not in self.tran_dic:
                    self.tran_dic[t]=v
                    v+=1
        self.tran_size=len(self.tran_dic)
        self.pos_size=len(self.pos_dic)

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

    def get_transaction(self,item,index):
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
                stack_word.append([item[int(j[0])-1][1],item[int(j[0])-1][4]])
            for j in range(k,len(item)):
                buffer_word.append([item[j][1],item[j][4]])
            if p<2:
                if len(item)==k and p==1:
                    transaction.append(['r',stack[0][2].decode()])
                    tran_list=copy(transaction)
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
                else:
                    break
            sample_list.append([stack_word,buffer_word,tran_list,index])
        return transaction,sample_list

    def get_parsing_item(self,transaction,sentence):
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

    def generate_batch(self,wv,parser):
        l=self.length
        vector_size=len(wv['unk'].tolist())
        zero=np.zeros(vector_size+self.pos_size,dtype=float).tolist()
        zero_t=np.zeros(self.tran_size,dtype=float).tolist()
        unk=wv['unk'].tolist()
        tran_embedding_one_hot=np.eye(self.tran_size).tolist()
        pos_embedding_one_hot=np.eye(self.pos_size).tolist()
        tran_embedding=parser.G.data.cpu().numpy().transpose().tolist()
        pin=self.sample_pin
        batch=list()
        stack_square=list()
        buffer_square=list()
        tran_square=list()
        label_square=list()
        stack_len=0
        buffer_len=0
        tran_len=0
        batch_size=0
        while(self.sample_list[pin][3]==self.sentence_index):
            if pin>=l:
                pin=pin-l
                self.sample_period+=1
            sample_list=self.sample_list[pin]
            pin+=1
            l0=len(sample_list[0])
            l1=len(sample_list[1])
            l2=len(sample_list[2])-1
            if l0>stack_len:
                stack_len=l0
            if l1>buffer_len:
                buffer_len=l1
            if l2>tran_len:
                tran_len=l2
            tran=list()
            for word in sample_list[2]:
                if len(word)==2:
                    word=word[0]+word[1]
                tran.append(word)
            label=tran[len(tran)-1]
            tran.pop()
            stack_square.append(sample_list[0])
            buffer_square.append(sample_list[1])
            tran_square.append(tran)
            label_square.append(label)
            batch_size+=1

        for i in range(len(stack_square)):
            if len(stack_square[i])<stack_len:
                temp=stack_square[i]
                stack_square[i]=np.zeros(stack_len-len(stack_square[i]),dtype=int).tolist()
                stack_square[i].extend(temp)
        for i in range(len(buffer_square)):
            if len(buffer_square[i])<buffer_len:
                temp=buffer_square[i]
                buffer_square[i]=np.zeros(buffer_len-len(buffer_square[i]),dtype=int).tolist()
                buffer_square[i].extend(temp)
        for i in range(len(tran_square)):
            if len(tran_square[i])<tran_len:
                temp=tran_square[i]
                tran_square[i]=np.zeros(tran_len-len(tran_square[i]),dtype=int).tolist()
                tran_square[i].extend(temp)

        stack_rev=np.zeros((stack_len,batch_size),dtype=int).tolist()
        buffer_rev=np.zeros((buffer_len,batch_size),dtype=int).tolist()
        tran_rev=np.zeros((tran_len,batch_size),dtype=int).tolist()
        for i in range(stack_len):
            for j in range(batch_size):
                if isinstance(stack_square[j][i],int):
                    stack_rev[i][j]=zero
                elif stack_square[j][i][0].decode('utf-8','replace').lower() in wv:
                    stack_rev[i][j]=wv[stack_square[j][i][0].decode('utf-8','replace').lower()].tolist()+pos_embedding_one_hot[self.pos_dic[stack_square[j][i][1]]]
                else:
                    stack_rev[i][j]=unk+pos_embedding_one_hot[self.pos_dic[stack_square[j][i][1]]]
        for i in range(buffer_len):
            for j in range(batch_size):
                if isinstance(buffer_square[j][i],int):
                    buffer_rev[i][j]=zero
                elif buffer_square[j][i][0].decode('utf-8','replace').lower() in wv:
                    buffer_rev[i][j]=wv[buffer_square[j][i][0].decode('utf-8','replace').lower()].tolist()+pos_embedding_one_hot[self.pos_dic[buffer_square[j][i][1]]]
                else:
                    buffer_rev[i][j]=unk+pos_embedding_one_hot[self.pos_dic[buffer_square[j][i][1]]]
        for i in range(tran_len):
            for j in range(batch_size):
                if isinstance(tran_square[j][i],int):
                    tran_rev[i][j]=zero_t
                else:
                    tran_rev[i][j]=tran_embedding[self.tran_dic[tran_square[j][i]]]
        for i in range(batch_size):
            label_square[i]=tran_embedding_one_hot[self.tran_dic[label_square[i]]]
        self.sample_pin=pin
        batch=[stack_rev,buffer_rev,tran_rev,label_square]
        bs=batch_size
        batch_size=0
        self.sentence_index+=1
        if self.sentence_index>=self.total_sentence:
            self.sentence_index=0
        return batch,bs

    def parser(self,sentence,parser,wv):
        vector_size=len(wv['unk'])
        stack=list()
        buffer=list()
        tran=list()
        buffer=sentence.lower().split()
        tran_embedding=np.eye(self.tran_size).tolist()
        k=0
        for i in range(len(buffer)*2):
            if len(stack)==0:
                stack_in=np.zeros([1,1,vector_size],dtype=float).tolist()
            else:
                stack_in=np.zeros([len(stack),1],dtype=float).tolist()
            if len(buffer)==0:
                buffer_in=np.zeros([1,1,vector_size],dtype=float).tolist()
            else:
                buffer_in=np.zeros([len(buffer),1],dtype=float).tolist()
            if len(tran)==0:
                tran_in=np.zeros([1,1,self.tran_size],dtype=float).tolist()
            else:
                tran_in=np.zeros([len(tran),1],dtype=float).tolist()
            for i in range(len(stack)):
                stack_in[i][0]=wv[stack[i]].tolist()
            for i in range(len(buffer)):
                buffer_in[i][0]=wv[buffer[i]].tolist()
            for i in range(len(tran)):
                tran_in[i][0]=tran_embedding[self.tran_dic[tran[i]]]
            import torch
            stack_in=torch.autograd.Variable(torch.Tensor(stack_in).cuda())
            buffer_in=torch.autograd.Variable(torch.Tensor(buffer_in).cuda())
            tran_in=torch.autograd.Variable(torch.Tensor(tran_in).cuda())
            result=parser(stack_in,buffer_in,tran_in).data.cpu().numpy().tolist()[0]
            temp=0
            for i in range(len(result)):
                if result[i]>result[temp]:
                    temp=i
            tran_item=self.tran_dic_rev[temp]
            if tran_item=='shift' and k<len(buffer):
                stack.append(buffer[k])
                k+=1
                tran.append('shift')
            elif tran_item[0]=='r' and len(stack)>0:
                del stack[len(stack)-1]
                tran.append(tran_item)
            elif tran_item[0]=='l' and len(stack)>1:
                del stack[len(stack)-2]
                tran.append(tran_item)
            else:
                tran.append(tran_item)
        return tran










