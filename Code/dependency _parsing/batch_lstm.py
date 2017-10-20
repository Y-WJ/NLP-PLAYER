# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:19:52 2017

@author: 颜
"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import relu
from torch.nn.functional import softmax

class LSTM(torch.nn.Module):
    def __init__(self,size_input:int,size_output:int,batch_size:int):
        super(LSTM, self).__init__()

        self.size_x=size_input
        self.size_h=size_output
        self.batch_size=batch_size

        self.i=Variable(torch.zeros([batch_size,self.size_h]).cuda())
        self.f=Variable(torch.zeros([batch_size,self.size_h]).cuda())
        self.c=Variable(torch.zeros([batch_size,self.size_h]).cuda())
        self.c_1=Variable(torch.zeros([batch_size,self.size_h]).cuda())
        self.h=Variable(torch.zeros([batch_size,self.size_h]).cuda())
        self.h_1=Variable(torch.zeros([batch_size,self.size_h]).cuda())
        self.o=Variable(torch.zeros([batch_size,self.size_h]).cuda())

        self.W_ix=Parameter(torch.normal(torch.zeros([self.size_x,self.size_h]),0.1))
        self.W_ih=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_ic=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_fx=Parameter(torch.normal(torch.zeros([self.size_x,self.size_h]),0.1))
        self.W_fh=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_fc=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_cx=Parameter(torch.normal(torch.zeros([self.size_x,self.size_h]),0.1))
        self.W_ch=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_ox=Parameter(torch.normal(torch.zeros([self.size_x,self.size_h]),0.1))
        self.W_oh=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_oc=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.b_i=Parameter(torch.zeros([self.size_h]))
        self.b_f=Parameter(torch.zeros([batch_size,self.size_h]))
        self.b_c=Parameter(torch.zeros([self.size_h]))
        self.b_o=Parameter(torch.zeros([self.size_h]))

    def forward(self,iterator):
        self.reset()
        if iterator.size()==torch.Size([]):
            return self.h
        for x in iterator:
            self.h_1=self.h
            self.i=torch.sigmoid(torch.matmul(x,self.W_ix)+torch.matmul(self.h_1,self.W_ih)+torch.matmul(self.c_1,self.W_ic)+self.b_i)
            self.f=torch.sigmoid(torch.matmul(x,self.W_fx)+torch.matmul(self.h_1,self.W_fh)+torch.matmul(self.c_1,self.W_fc)+self.b_f)
            self.c=self.f*self.c_1+self.i*torch.tanh(torch.matmul(x,self.W_cx)+torch.matmul(self.h_1,self.W_ch)+self.b_c)
            self.o=torch.sigmoid(torch.matmul(x,self.W_ox)+torch.matmul(self.h_1,self.W_oh)+torch.matmul(self.c,self.W_oc)+self.b_o)
            self.h=self.o*torch.tanh(self.c)
            self.c_1=self.c
        return self.h

    def reset(self):
        self.i=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())
        self.f=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())
        self.c=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())
        self.c_1=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())
        self.h=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())
        self.h_1=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())
        self.o=Variable(torch.zeros([self.batch_size,self.size_h]).cuda())


class Parsing(torch.nn.Module):
    def __init__(self,size_vector:int,size_tran:int,size_out:int,batch_size:int):
        super(Parsing, self).__init__()

        self.size_vector=size_vector
        self.size_tran=size_tran
        self.size_out=size_out

        self.stack_lstm=LSTM(size_vector,size_out,batch_size)
        self.buffer_lstm=LSTM(size_vector,size_out,batch_size)
        self.tran_lstm=LSTM(size_tran,size_out,batch_size)

        self.W=Parameter(torch.normal(torch.zeros([self.size_out*3,self.size_out]),0.1))
        self.G=Parameter(torch.normal(torch.zeros([self.size_out,self.size_tran]),0.1))
        self.b_W=Parameter(torch.zeros([self.size_out]))
        self.b_G=Parameter(torch.zeros([self.size_tran]))

    def forward(self,s,b,t):
        stack=self.stack_lstm(s)
        buffer=self.buffer_lstm(b)
        tran=self.tran_lstm(t)
        p=relu(torch.matmul(torch.cat((stack,buffer,tran),1),self.W)+self.b_W)
        out=softmax(torch.matmul(p,self.G)+self.b_G)
        return out