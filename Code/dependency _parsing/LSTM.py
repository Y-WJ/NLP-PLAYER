# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:46:33 2017

@author: 颜
"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import relu
from torch.nn.functional import softmax

class LSTM(torch.nn.Module):
    def __init__(self,size_input:int,size_output:int):
        super(LSTM, self).__init__()

        self.size_x=size_input
        self.size_h=size_output

        self.i=Variable(torch.zeros([self.size_h]))
        self.f=Variable(torch.zeros([self.size_h]))
        self.c=Variable(torch.zeros([self.size_h]))
        self.c_1=Variable(torch.zeros([self.size_h]))
        self.h=Variable(torch.zeros([self.size_h]))
        self.h_1=Variable(torch.zeros([self.size_h]))
        self.o=Variable(torch.zeros([self.size_h]))

        self.W_ix=Parameter(torch.normal(torch.zeros([self.size_h,self.size_x]),0.1))
        self.W_ih=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_ic=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_fx=Parameter(torch.normal(torch.zeros([self.size_h,self.size_x]),0.1))
        self.W_fh=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_fc=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_cx=Parameter(torch.normal(torch.zeros([self.size_h,self.size_x]),0.1))
        self.W_ch=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_ox=Parameter(torch.normal(torch.zeros([self.size_h,self.size_x]),0.1))
        self.W_oh=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.W_oc=Parameter(torch.normal(torch.zeros([self.size_h,self.size_h]),0.1))
        self.b_i=Parameter(torch.zeros([self.size_h]))
        self.b_f=Parameter(torch.zeros([self.size_h]))
        self.b_c=Parameter(torch.zeros([self.size_h]))
        self.b_o=Parameter(torch.zeros([self.size_h]))

    def forward(self,iterator:iter):
        self.reset()
        for x in iterator:
            self.h_1=self.h
            self.i=torch.sigmoid(torch.matmul(self.W_ix,x)+torch.matmul(self.W_ih,self.h_1)+torch.matmul(self.W_ic,self.c_1)+self.b_i)
            self.f=torch.sigmoid(torch.matmul(self.W_fx,x)+torch.matmul(self.W_fh,self.h_1)+torch.matmul(self.W_fc,self.c_1)+self.b_f)
            self.c=self.f*self.c_1+self.i*torch.tanh(torch.matmul(self.W_cx,x)+torch.matmul(self.W_ch,self.h_1)+self.b_c)
            self.o=torch.sigmoid(torch.matmul(self.W_ox,x)+torch.matmul(self.W_oh,self.h_1)+torch.matmul(self.W_oc,self.c)+self.b_o)
            self.h=self.o*torch.tanh(self.c)
            self.c_1=self.c
        return self.h

    def reset(self):
        self.i=Variable(torch.zeros([self.size_h]))
        self.f=Variable(torch.zeros([self.size_h]))
        self.c=Variable(torch.zeros([self.size_h]))
        self.c_1=Variable(torch.zeros([self.size_h]))
        self.h=Variable(torch.zeros([self.size_h]))
        self.h_1=Variable(torch.zeros([self.size_h]))
        self.o=Variable(torch.zeros([self.size_h]))


class Parsing(torch.nn.Module):
    def __init__(self,size_vector:int,size_tran:int,size_out:int):
        super(Parsing, self).__init__()

        self.size_vector=size_vector
        self.size_tran=size_tran
        self.size_out=size_out

        self.stack_lstm=LSTM(size_vector,size_out)
        self.buffer_lstm=LSTM(size_vector,size_out)
        self.tran_lstm=LSTM(size_tran,size_out)

        self.W=Parameter(torch.normal(torch.zeros([self.size_tran,self.size_out*3]),0.1))
        self.G=Parameter(torch.normal(torch.zeros([self.size_tran,self.size_tran]),0.1))
        self.b_W=Parameter(torch.zeros([self.size_tran]))
        self.b_G=Parameter(torch.zeros([self.size_tran]))

    def forward(self,s,b,t):
        self.stack_lstm.reset()
        self.buffer_lstm.reset()
        self.tran_lstm.reset()
        stack=self.stack_lstm(s)
        buffer=self.buffer_lstm(b)
        tran=self.tran_lstm(t)
        p=relu(torch.matmul(self.W,torch.cat((stack,buffer,tran),0))+self.b_W)
        out=softmax(torch.matmul(p,self.G)+self.b_G)
        return out



'''
batch_size=2
vector_size=5
tran_size=5
out_size=4
sequence=5
'''
par=Parsing(5,5,4)
s=Variable(torch.randn([4,5]))
b=Variable(torch.randn([5,5]))
t=Variable(torch.randn([2,5]))
out=par(s,b,t)

'''
lis=Variable(torch.randn([batch_size,sequence,vector_size]))
lenth=[5,3]
s=pack_padded_sequence(lis,lenth,batch_first=True)
out,_=par.stack_lstm(s)
out=pad_packed_sequence(out,batch_first=True)
print(lis)
print(out)
o=Variable(torch.randn([batch_size,out_size]))

print(o)
'''



