# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:31:59 2017

@author: 颜
"""

import CONLL as con
import batch_lstm as lstm
import torch
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import gensim

#---------------------------参数设置------------------------------------
train_step=90000                       #建议与batch_size组合能遍历sample_list两遍以上，即train_step*batch_size>con.lenth*2
vector_size=300                         #等于你所使用的词向量维度
out_size=vector_size*2                  #建议大于vector_size*2+tran_size
check_point=10                         #检查点
#----------------------加载COMLLU文件以及词向量字典---------------------

conll=con.CONLL('train.conllu')
wv=gensim.models.KeyedVectors.load_word2vec_format("wiki_gen_vector\\wv.bin",binary=True)

#-------------------------声明网络和优化器------------------------------
parsing=lstm.Parsing(vector_size+conll.pos_size,conll.tran_size,out_size)
parsing.cuda()
optimizer = optim.Adadelta(parsing.parameters())

#------------------------设置监视器-------------------------------------
t=tqdm(range(train_step),desc="Training",ncols=130,ascii=True,unit="Batch")
info={'INIT_LOSS':0.,'LOSS':0.,'period':0.}

#------------------------GGG!!!----------------------------------------

LOSS=0
for i in t:
    parsing.zero_grad()
    [stack,buffer,tran,label],batch_size=conll.generate_batch(wv,parsing)
    stack=Variable(torch.Tensor(stack).cuda())
    buffer=Variable(torch.Tensor(buffer).cuda())
    tran=Variable(torch.Tensor(tran).cuda())
    label=Variable(torch.Tensor(label).cuda())
    out=parsing(stack,buffer,tran,batch_size)
    loss=-torch.sum(torch.log(torch.sum(out*label,1)))/batch_size
    loss.backward()
    LOSS+=loss
    optimizer.step()
    if i==0:
        info['INIT_LOSS']=LOSS.cpu().data.numpy()[0]
        t.set_postfix(**info)
    if i%check_point==0:
        info['LOSS']=LOSS.cpu().data.numpy()[0]/check_point
        info['period']=conll.sample_period
        LOSS=0
        t.set_postfix(**info)
print("FINISHED!!!")
torch.save(parsing,"parsing.pkl")

