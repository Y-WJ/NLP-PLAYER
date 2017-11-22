<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:37:21 2017

@author: 颜
"""
'''
Thie file defined YOLO_network based on pytorch
All default parameters same as which in YOLO-paper are given in class parameter
The whole network still adjust to your custom settings
'''

import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import pickle
import os

class YOLO(torch.nn.Module):
    def __init__(self,para):
        super(YOLO,self).__init__()
        self.para=para
        self.verify_para(para)
        kernel_0=para.S                     #7
        padding_0=para.S//2                 #3
        kernel_1=kernel_0//2                #3
        padding_1=kernel_1//2               #1
        kernel_2=1
        offset=np.transpose(np.reshape(np.array([np.arange(para.S,dtype=float)]*para.S *para.B),(para.B,para.S,para.S)),(1,2,0))
        self.offset=Variable(torch.Tensor(offset)).resize(1,self.para.S,self.para.S,self.para.B).repeat(self.para.batch_size,1,1,1)
        self.tensor_0=Variable(torch.Tensor([0]))
        self.tensor_1=Variable(torch.Tensor([1e-10]))
        self.tensor_2=Variable(torch.ones([para.batch_size,para.S,para.S,para.B]))
        self.tensor_type='torch.FloatTensor'
        self.bound_0=self.para.S*self.para.S*self.para.label_nums
        self.bound_1=self.bound_0+self.para.S*self.para.S*self.para.B
        if para.USING_GPU:
            self.offset=self.offset.cuda()
            self.tensor_0=self.tensor_0.cuda()
            self.tensor_1=self.tensor_1.cuda()
            self.tensor_2=self.tensor_2.cuda()
            self.tensor_type='torch.cuda.FloatTensor'
        self.predict_vec_size=para.B*5+para.label_nums
#-----------------各层参数-------------------------------
#block_1:
        self.conv_1=nn.Conv2d(para.image_channel,64,kernel_0,stride=2,padding=padding_0)
#block_2:
        self.conv_2=nn.Conv2d(64,192,kernel_1,padding=padding_1)
#block_3:
        self.conv_3=nn.Conv2d(192,128,kernel_2)
        self.conv_4=nn.Conv2d(128,256,kernel_1,padding=padding_1)
        self.conv_5=nn.Conv2d(256,256,kernel_2)
        self.conv_6=nn.Conv2d(256,512,kernel_1,padding=padding_1)
#block_4:
        self.conv_7=nn.Conv2d(512,256,kernel_2)
        self.conv_8=nn.Conv2d(256,512,kernel_1,padding=padding_1)
        self.conv_9=nn.Conv2d(512,256,kernel_2)
        self.conv_10=nn.Conv2d(256,512,kernel_1,padding=padding_1)
        self.conv_11=nn.Conv2d(512,256,kernel_2)
        self.conv_12=nn.Conv2d(256,512,kernel_1,padding=padding_1)
        self.conv_13=nn.Conv2d(512,256,kernel_2)
        self.conv_14=nn.Conv2d(256,512,kernel_1,padding=padding_1)
        self.conv_15=nn.Conv2d(512,512,kernel_2)
        self.conv_16=nn.Conv2d(512,1024,kernel_1,padding=padding_1)
#block_5:
        self.conv_17=nn.Conv2d(1024,512,kernel_2)
        self.conv_18=nn.Conv2d(512,1024,kernel_1,padding=padding_1)
        self.conv_19=nn.Conv2d(1024,512,kernel_2)
        self.conv_20=nn.Conv2d(512,1024,kernel_1,padding=padding_1)
        self.conv_21=nn.Conv2d(1024,1024,kernel_1,padding=padding_1)
        self.conv_22=nn.Conv2d(1024,1024,kernel_1,stride=2,padding=padding_1)
#block_6:
        self.conv_23=nn.Conv2d(1024,1024,kernel_1,padding=padding_1)
        self.conv_24=nn.Conv2d(1024,1024,kernel_1,padding=padding_1)
#block_7:
        self.conn_25=nn.Linear(para.S*para.S*1024,512)
        self.conn_26=nn.Linear(para.S*para.S*1024,para.S//2*1024)
        self.drop_27=nn.Dropout()
        self.conn_28=nn.Linear(para.S//2*1024,para.S*para.S*self.predict_vec_size)

    def verify_para(self,para):
        '''
        参数校验
        '''
        import os
        p=True
        if not para.image_size/para.S==64:
            p=False
        if para.S%2==0 or para.S//2%2==0:
            p=False
        if not p:
            print("PARAMETER INVALID!!!")
            os._exit

    def forward(self,x):
        '''
        前向网络
        '''
        x=x.transpose(2,3).transpose(1,2)
        x=func.max_pool2d(func.leaky_relu(self.conv_1(x),0.1),[2,2],stride=2)
        x=func.max_pool2d(func.leaky_relu(self.conv_2(x),0.1),[2,2],stride=2)
        x=func.leaky_relu(self.conv_3(x),0.1)
        x=func.leaky_relu(self.conv_4(x),0.1)
        x=func.leaky_relu(self.conv_5(x),0.1)
        x=func.max_pool2d(func.relu(self.conv_6(x)),[2,2],stride=2)
        x=func.leaky_relu(self.conv_7(x),0.1)
        x=func.leaky_relu(self.conv_8(x),0.1)
        x=func.leaky_relu(self.conv_9(x),0.1)
        x=func.leaky_relu(self.conv_10(x),0.1)
        x=func.leaky_relu(self.conv_11(x),0.1)
        x=func.leaky_relu(self.conv_12(x),0.1)
        x=func.leaky_relu(self.conv_13(x),0.1)
        x=func.leaky_relu(self.conv_14(x),0.1)
        x=func.leaky_relu(self.conv_15(x),0.1)
        x=func.max_pool2d(func.leaky_relu(self.conv_16(x),0.1),[2,2],stride=2)
        x=func.leaky_relu(self.conv_17(x),0.1)
        x=func.leaky_relu(self.conv_18(x),0.1)
        x=func.leaky_relu(self.conv_19(x),0.1)
        x=func.leaky_relu(self.conv_20(x),0.1)
        x=func.leaky_relu(self.conv_21(x),0.1)
        x=func.leaky_relu(self.conv_22(x),0.1)
        x=func.leaky_relu(self.conv_23(x),0.1)
        x=func.leaky_relu(self.conv_24(x),0.1)
        x=x.resize(self.para.batch_size,7*7*1024)
        x=func.leaky_relu(self.conn_25(x),0.1)
        x=func.leaky_relu(self.conn_26(x),0.1)
        x=self.drop_27(x)
        x=self.conn_28(x)
        return x

    def IOU(self,box_1,box_2):
        '''
        计算两组box的IOU,对一对box来说，它们的IOU定义为重叠面积比上总面积
        box_1,box_2  5-D tensor (45,7,7,2,4)
        '''
        box_1=torch.cat(((box_1[:,:,:,:,0] - box_1[:,:,:,:,2]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                           (box_1[:,:,:,:,1] - box_1[:,:,:,:,3]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                           (box_1[:,:,:,:,0] + box_1[:,:,:,:,2]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                           (box_1[:,:,:,:,1] + box_1[:,:,:,:,3]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1)),4)
#cat后最高维为[左，下，右，上]四元组
        box_2=torch.cat(((box_2[:,:,:,:,0]-box_2[:,:,:,:,2]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                           (box_2[:,:,:,:,1]-box_2[:,:,:,:,3]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                           (box_2[:,:,:,:,0]+box_2[:,:,:,:,2]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                           (box_2[:,:,:,:,1]+box_2[:,:,:,:,3]/2.0).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1)),4)
#cat后最高维为[左，下，右，上]四元组

        bound_1=torch.max(box_1[:,:,:,:,:2],box_2[:,:,:,:,:2])#(45,7,7,2,2)
        bound_2=torch.min(box_1[:,:,:,:,2:],box_2[:,:,:,:,2:])#每个box对重叠部分的两角
        intersection=torch.max(bound_2-bound_1,self.tensor_0)             #计算重叠区域宽和高
        inter_area=intersection[:,:,:,:,0]*intersection[:,:,:,:,1]#两box重叠面积

        box_1_area=(box_1[:,:,:,:,2]-box_1[:,:,:,:,0])*(box_1[:,:,:,:,3]-box_1[:,:,:,:,1])#(47,7,7,2)
        box_2_area=(box_2[:,:,:,:,2]-box_2[:,:,:,:,0])*(box_2[:,:,:,:,3]-box_2[:,:,:,:,1])#每个box的面积

        IOU=inter_area/torch.max(box_1_area+box_2_area-inter_area,self.tensor_1)#IOU为重叠面积与总面积之比

        return torch.clamp(IOU,min=0.0,max=1.0)#以防万一

    def loss(self,predict,label):
        '''
        根据网络的输出predict,和generate_batch给出的标签计算损失
        predict 4D-tensor,shape=[batch_size,S,S,30]
        最后一维各个元素：
        [0:1]  每个grid有两个box,这两个元素分别描述这两个box的预测IOU
        [2:9]  两个box的坐标，每个坐标四个值，总共8个
        [10:29] 20个标签的预测概率

        label 4D-tensor,shape=[batch_size,S,S,25]
        最后一维各个元素：
        [0]    二值，是否有某个box中心落在当前grid，若有为1，否则为0
        [1:4]  box坐标，若[0]==1，这一项保存box的坐标,中心坐标为全图相对坐标，w,h为像素坐标
        [5:24] 二值类别标记，若标为第i个类别的box中心落在当前grid,则[5+i]=1，否则为0
        '''
        predict_class_pro=predict[:,:self.bound_0].resize(self.para.batch_size,self.para.S,self.para.S,self.para.label_nums)
        predict_IOU=predict[:,self.bound_0:self.bound_1].resize(self.para.batch_size,self.para.S,self.para.S,self.para.B)
        predict_box_grid=predict[:,self.bound_1:].resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,4)

        label_bool=label[:,:,:,0].resize(self.para.batch_size,self.para.S,self.para.S,1)#45,7,7,1
        label_box_image=label[:,:,:,1:5].resize(self.para.batch_size,self.para.S,self.para.S,1,4).repeat(1,1,1,self.para.B,1)/float(self.para.image_size)
        label_class_bool=label[:,:,:,5:]#45,7,7,20

        predict_box_image= torch.cat(
                (((predict_box_grid[:,:,:,:,0]+self.offset)/float(self.para.S)).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                 ((predict_box_grid[:,:,:,:,1]+self.offset.transpose(1,2))/float(self.para.S)).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                 torch.pow(predict_box_grid[:,:,:,:,2],2).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                 torch.pow(predict_box_grid[:,:,:,:,3],2).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1)
                 ),4)#(45,7,7,2,4)
        #预测结果是box中心在当前grid的相对坐标,转换为在整张图片的相对坐标,注意由于损失计算取w,h的平方根，这里的预测结果可能为负，因此把预测结果当作w,h的平方根，这里后两项求平方得到w,h
        IOU_predict_label=self.IOU(predict_box_image,label_box_image)#(45,7,7,2)

        obj_mask=torch.max(IOU_predict_label,3,keepdim=True)[0]#(45,7,7,1)
        obj_mask=(IOU_predict_label>=obj_mask).type(self.tensor_type)*label_bool#(45,7,7,2)
        noobj_mask=self.tensor_2-obj_mask#(45,7,7,2)
        coord_mask=obj_mask.resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1)#45,7,7,2,1
        #每个grid的每个box如果用于预测某个类，那么对应的obj_mask为1，否则为0
        #noobj_mask正好相反

        label_box_grid=torch.cat(
                ((label_box_image[:,:,:,:,0]*float(self.para.S)-self.offset).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                 (label_box_image[:,:,:,:,1]*float(self.para.S)-self.offset.transpose(1,2)).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                 torch.sqrt(label_box_image[:, :, :, :, 2]).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1),
                 torch.sqrt(label_box_image[:, :, :, :, 3]).resize(self.para.batch_size,self.para.S,self.para.S,self.para.B,1)
                 ),4)#(45,7,7,2,4)
        #标签为在整张图片的相对坐标，转换为在当前grid的相对坐标，标签的w,h都是正值，可以放心开根号

        box_loss=torch.sum(torch.pow(coord_mask*(predict_box_grid-label_box_grid),2))*self.para.coord/self.para.batch_size
        obj_loss=torch.sum(torch.pow(obj_mask*(predict_IOU-IOU_predict_label),2))/self.para.batch_size
        noobj_loss=torch.sum(torch.pow(noobj_mask*predict_IOU,2))*self.para.noobj/self.para.batch_size
        class_loss=torch.sum(torch.pow(label_bool*(predict_class_pro-label_class_bool),2))/self.para.batch_size
        #根据原文公式计算四项损失，其中box_loss，noobj_loss用原文给出的超参修正

        return class_loss+obj_loss+noobj_loss+box_loss

    def load_weight(self):
        if os.path.isfile(self.para.PRETRAINED_WEIGHT_FILE):
            with open(self.para.PRETRAINED_WEIGHT_FILE,'rb') as f:
                weight=pickle.load(f)
        else:
            print("NO WEIGHT FILE FOUNDED!!!")
            os._exit
        if not len(weight)==54:
            print("INVALID WEIGHT FILE!!!")
            os._exit
        i=0
        for para in self.parameters():
            if len(np.shape(weight[i]))==4:
                w=np.transpose(weight[i],(3,2,0,1))
            elif len(np.shape(weight[i]))==2:
                w=np.transpose(weight[i],(1,0))
            else:
                w=weight[i]
            para.data=torch.from_numpy(w)
            i+=1