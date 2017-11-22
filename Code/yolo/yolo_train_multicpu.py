<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:23:38 2017

@author: 颜
"""

import torch
import config
from torch.autograd.variable import Variable
from tqdm import tqdm
#from config import init_weight
from yolo_net import YOLO
from voc import pascal_voc
import torch.multiprocessing as mp
import os


def train(yolo,para,voc,rank):
    optimizer=torch.optim.SGD(yolo.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)
    LOSS=0
    t=tqdm(range(para.train_step),desc="Trainer"+str(rank),ncols=110,ascii=True,unit="Batches",position=rank)
    info={'INIT_LOSS':0.,'LOSS':0.,'epoch':0.}
    for i in t:
        if (voc.cursor-rank)%para.CPU_PROCESSING==0:
            yolo.zero_grad()
            images,labels=voc.generate_batch()
            images=Variable(torch.Tensor(images))
            labels=Variable(torch.Tensor(labels))
            if para.USING_GPU:
                images=images.cuda()
                labels=labels.cuda()
            predict=yolo.forward(images)
            loss=yolo.loss(predict,labels)
            loss.backward()
            LOSS+=loss.cpu().data[0]
            optimizer.step()
            if i==rank:
                info['INIT_LOSS']=LOSS
                t.set_postfix(**info)
            elif i%para.check_point==0:
                info['LOSS']=LOSS/para.check_point
                info['epoch']=voc.epoch
                LOSS=0
                t.set_postfix(**info)
            if i%para.save_point==0 and not i==0:
                torch.save(yolo,para.MODULE_NAME)
        else:
            voc.cursor+=1

if __name__=='__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    para=config.Parameter
    voc=pascal_voc(para)
    yolo=YOLO(para)
    if para.USING_GPU:
        yolo.cuda()
    #yolo.apply(init_weight)
    processes = []
    for rank in range(para.CPU_PROCESSING):
        p = mp.Process(target=train, args=(yolo,para,voc,rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()