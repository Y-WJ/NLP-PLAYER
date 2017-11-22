<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:08:19 2017

@author: 颜
"""
'''
This file define
'''
#common parameter
#import torch.nn as nn
class Parameter():
    '''
    USINGGPU:是否使用GPU训练
    CPU_PROCESSING:指定cpu并发进程数，在USINGGPU=0时有效
    UPDATE_MODEL:训练或推断，后者不更新模型
    REBUILD:更新图像缓存
    LOAD_MODEL:从指定文件加载模型用于训练或推断
    SAVE_MODEL:训练中模型保存至指定文件
    '''
    train_step=20000
    check_point=10
    save_point=1000
    batch_size=45
    image_size=448                                  #s.t.:image_size/s=64
    image_channel=3
    label_nums=20
    S=7                                             #每行/列grid数量,限定为奇数
    B=2                                             #每个grid预测box数量
    coord=5
    noobj=0.5
    THRESHOLD=0.35
    IOU_THRESHOLD=0.2
#-----------------init voc------------------------------------
    USING_GPU=False                                 #使用GPU训练
    CPU_PROCESSING=3                                #cpu进程数，在USING_GPU=0时有效
    UPDATE_MODEL=True                               #训练或者推断，后者不更新模型
    REBUILD=False                                   #更新训练数据缓存
    USING_PRETRAINED_WEIGHT=True
    PRETRAINED_WEIGHT_FILE='weight.pkl'
    MODEL_PATH='model'
    LOAD_MODEL='YOLO_45_20000.pkl'                  #从指定文件载入模型
    SAVE_MODEL='YOLO_'+str(batch_size)+'_'+str(train_step)+'.pkl'
    VOC_PATH=''
    VOC_VERSION='VOC2012'
    CACHE_PATH='cache'
    DATA_PATH='D:/1_work/DATA_SET/object-detection-crowdai'
    PHASE='train'
    CLASSES=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    def __init__(self):
        return
'''
def init_weight(m):
    if type(m)==nn.Conv2d:
        nn.init.uniform(m.weight.data,a=0,b=2)
    elif type(m)==nn.Linear:
        nn.init.normal(m.weight.data,mean=0,std=0.1)

