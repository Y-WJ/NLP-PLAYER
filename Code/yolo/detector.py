<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:38:17 2017

@author: 颜
"""

import numpy as np
import os
import cv2
import torch
from torch.autograd.variable import Variable
from config import Parameter


class Detector(object):
    def __init__(self,para:Parameter):
        self.para=para
        self.yolo=torch.load(os.path.join(para.MODEL_PATH,para.LOAD_MODEL))
        self.yolo.para.batch_size=1
        self.offset=torch.Tensor(np.transpose(np.reshape(np.array([np.arange(para.S,dtype=float)]*para.S *para.B),(para.B,para.S,para.S)),(1,2,0)))
        self.offset=Variable(self.offset)
        if para.USING_GPU:
            self.yolo=self.yolo.cuda()
            self.offset=self.offset.cuda()
        else:
            self.yolo=self.yolo.cpu()
        self.threshold=para.THRESHOLD
        self.iou_threshold=para.IOU_THRESHOLD
        self.bound_0=self.para.S*self.para.S*self.para.label_nums
        self.bound_1=self.bound_0+self.para.S*self.para.S*self.para.B

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, result[i][0] +' : %.2f' % result[i][5],(x-w+5,y-h-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.CV_16U)

    def detect(self,image):
        img_h,img_w,_=image.shape
        inputs=cv2.resize(image,(448,448))
        inputs=cv2.cvtColor(inputs,cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs=(inputs/255.0)*2.0-1.0
        inputs=np.reshape(inputs,(1,448,448,3))

        result=self.detect_from_yolo(inputs)[0]

        for i in range(len(result)):
            result[i][1]*=(1.0*img_w/self.para.image_size)
            result[i][2]*=(1.0*img_h/self.para.image_size)
            result[i][3]*=(1.0*img_w/self.para.image_size)
            result[i][4]*=(1.0*img_h/self.para.image_size)

        return result

    def detect_from_yolo(self, inputs):
        inputs=Variable(torch.Tensor(inputs))
        if self.para.USING_GPU:
            inputs=inputs.cuda()
        output=self.yolo(inputs)
        results=[]
        for i in range(output.size()[0]):
            results.append(self.interpret_output(output[i]))
        return results

    def interpret_output(self, predict):
        predict_class_pro=predict[:self.bound_0].resize(self.para.S,self.para.S,self.para.label_nums)
        predict_IOU=predict[self.bound_0:self.bound_1].resize(self.para.S,self.para.S,self.para.B)
        predict_box_grid=predict[self.bound_1:].resize(self.para.S,self.para.S,self.para.B,4)
        '''
        predict_box_grid=predict[:,:,self.para.B:self.para.B*5].cpu().numpy()
        predict_box_grid=predict_box_grid.reshape(7,7,2,4)
        predict_box_grid=torch.Tensor(predict_box_grid).cuda()
        '''
        predict_box_image= torch.cat(
                (((predict_box_grid[:,:,:,0]+self.offset)/float(self.para.S)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1),
                 ((predict_box_grid[:,:,:,1]+self.offset.transpose(0,1))/float(self.para.S)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1),
                 (torch.pow(predict_box_grid[:,:,:,2],2)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1),
                 (torch.pow(predict_box_grid[:,:,:,3],2)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1)
                 ),3)

        prob=(predict_IOU.resize(self.para.S,self.para.S,self.para.B,1).expand(self.para.S,self.para.S,self.para.B,self.para.label_nums))*(predict_class_pro.resize(self.para.S,self.para.S,1,self.para.label_nums).expand(self.para.S,self.para.S,self.para.B,self.para.label_nums))

        '''
        prob = np.zeros((self.para.S,self.para.S,self.para.B,self.para.label_nums))
        for i in range(self.para.B):
            for j in range(self.para.label_nums):
                prob[:, :, i, j] = np.multiply(predict_class_pro.cpu().numpy()[:, :, j], predict_IOU.cpu().numpy()[:, :, i])
        prob=torch.Tensor(prob).cuda()
        '''
        prob=prob.cpu().data.numpy()
        predict_box_image=predict_box_image.cpu().data.numpy()

        filter_mat_prob=np.array(prob>=self.threshold,dtype='bool')#shape=7x7x2x20
        filter_mat_boxe=np.nonzero(filter_mat_prob)
        boxe_filtered=predict_box_image[filter_mat_boxe[0],filter_mat_boxe[1], filter_mat_boxe[2]]#shape=nx4
        prob_filtered=prob[filter_mat_prob]#size=n
        classes_num_filtered=np.argmax(filter_mat_prob, axis=3)[filter_mat_boxe[0], filter_mat_boxe[1],filter_mat_boxe[2]]#size=n

        argsort=np.array(np.argsort(prob_filtered))[::-1]
        boxe_filtered=boxe_filtered[argsort]#size=nx4
        prob_filtered=prob_filtered[argsort]#size=n
        classes_num_filtered = classes_num_filtered[argsort]#size=n

        for i in range(len(boxe_filtered)):
            if prob_filtered.data[i]==0:
                continue
            for j in range(i+1,len(boxe_filtered)):
                if self.iou(boxe_filtered[i],boxe_filtered[j])>self.iou_threshold:
                    prob_filtered[j]=0.0

        filter_iou=np.array(prob_filtered > 0.0, dtype='bool')
        boxe_filtered=boxe_filtered[filter_iou]
        prob_filtered=prob_filtered[filter_iou]
        classes_num_filtered=classes_num_filtered[filter_iou]

        result=[]
        for i in range(len(boxe_filtered)):
            result.append([self.para.CLASSES[classes_num_filtered[i]], boxe_filtered[i][0], boxe_filtered[i][1], boxe_filtered[i][2], boxe_filtered[i][3], prob_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb=min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
        lr=min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection=tb*lr
        return intersection/(box1[2]*box1[3]+box2[2]*box2[3]-intersection)

    def image_detector(self, image, wait=0):
        #image=cv2.imread(imname)

        result=self.detect(image)

        self.draw_result(image, result)
        image=cv2.resize(image,(600,600))
        return image
        #cv2.imshow('Image', image)
        #cv2.waitKey(wait)

para=Parameter
D=Detector(para)
=======
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:38:17 2017

@author: 颜
"""

import numpy as np
import os
import cv2
import torch
from torch.autograd.variable import Variable
from config import Parameter


class Detector(object):
    def __init__(self,para:Parameter):
        self.para=para
        self.yolo=torch.load(os.path.join(para.MODEL_PATH,para.LOAD_MODEL))
        self.yolo.para.batch_size=1
        self.offset=torch.Tensor(np.transpose(np.reshape(np.array([np.arange(para.S,dtype=float)]*para.S *para.B),(para.B,para.S,para.S)),(1,2,0)))
        self.offset=Variable(self.offset)
        if para.USING_GPU:
            self.yolo=self.yolo.cuda()
            self.offset=self.offset.cuda()
        else:
            self.yolo=self.yolo.cpu()
        self.threshold=para.THRESHOLD
        self.iou_threshold=para.IOU_THRESHOLD
        self.bound_0=self.para.S*self.para.S*self.para.label_nums
        self.bound_1=self.bound_0+self.para.S*self.para.S*self.para.B

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, result[i][0] +' : %.2f' % result[i][5],(x-w+5,y-h-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.CV_16U)

    def detect(self,image):
        img_h,img_w,_=image.shape
        inputs=cv2.resize(image,(448,448))
        inputs=cv2.cvtColor(inputs,cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs=(inputs/255.0)*2.0-1.0
        inputs=np.reshape(inputs,(1,448,448,3))

        result=self.detect_from_yolo(inputs)[0]

        for i in range(len(result)):
            result[i][1]*=(1.0*img_w/self.para.image_size)
            result[i][2]*=(1.0*img_h/self.para.image_size)
            result[i][3]*=(1.0*img_w/self.para.image_size)
            result[i][4]*=(1.0*img_h/self.para.image_size)

        return result

    def detect_from_yolo(self, inputs):
        inputs=Variable(torch.Tensor(inputs))
        if self.para.USING_GPU:
            inputs=inputs.cuda()
        output=self.yolo(inputs)
        results=[]
        for i in range(output.size()[0]):
            results.append(self.interpret_output(output[i]))
        return results

    def interpret_output(self, predict):
        predict_class_pro=predict[:self.bound_0].resize(self.para.S,self.para.S,self.para.label_nums)
        predict_IOU=predict[self.bound_0:self.bound_1].resize(self.para.S,self.para.S,self.para.B)
        predict_box_grid=predict[self.bound_1:].resize(self.para.S,self.para.S,self.para.B,4)
        '''
        predict_box_grid=predict[:,:,self.para.B:self.para.B*5].cpu().numpy()
        predict_box_grid=predict_box_grid.reshape(7,7,2,4)
        predict_box_grid=torch.Tensor(predict_box_grid).cuda()
        '''
        predict_box_image= torch.cat(
                (((predict_box_grid[:,:,:,0]+self.offset)/float(self.para.S)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1),
                 ((predict_box_grid[:,:,:,1]+self.offset.transpose(0,1))/float(self.para.S)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1),
                 (torch.pow(predict_box_grid[:,:,:,2],2)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1),
                 (torch.pow(predict_box_grid[:,:,:,3],2)*float(self.para.image_size)).resize(self.para.S,self.para.S,self.para.B,1)
                 ),3)

        prob=(predict_IOU.resize(self.para.S,self.para.S,self.para.B,1).expand(self.para.S,self.para.S,self.para.B,self.para.label_nums))*(predict_class_pro.resize(self.para.S,self.para.S,1,self.para.label_nums).expand(self.para.S,self.para.S,self.para.B,self.para.label_nums))

        '''
        prob = np.zeros((self.para.S,self.para.S,self.para.B,self.para.label_nums))
        for i in range(self.para.B):
            for j in range(self.para.label_nums):
                prob[:, :, i, j] = np.multiply(predict_class_pro.cpu().numpy()[:, :, j], predict_IOU.cpu().numpy()[:, :, i])
        prob=torch.Tensor(prob).cuda()
        '''
        prob=prob.cpu().data.numpy()
        predict_box_image=predict_box_image.cpu().data.numpy()

        filter_mat_prob=np.array(prob>=self.threshold,dtype='bool')#shape=7x7x2x20
        filter_mat_boxe=np.nonzero(filter_mat_prob)
        boxe_filtered=predict_box_image[filter_mat_boxe[0],filter_mat_boxe[1], filter_mat_boxe[2]]#shape=nx4
        prob_filtered=prob[filter_mat_prob]#size=n
        classes_num_filtered=np.argmax(filter_mat_prob, axis=3)[filter_mat_boxe[0], filter_mat_boxe[1],filter_mat_boxe[2]]#size=n

        argsort=np.array(np.argsort(prob_filtered))[::-1]
        boxe_filtered=boxe_filtered[argsort]#size=nx4
        prob_filtered=prob_filtered[argsort]#size=n
        classes_num_filtered = classes_num_filtered[argsort]#size=n

        for i in range(len(boxe_filtered)):
            if prob_filtered.data[i]==0:
                continue
            for j in range(i+1,len(boxe_filtered)):
                if self.iou(boxe_filtered[i],boxe_filtered[j])>self.iou_threshold:
                    prob_filtered[j]=0.0

        filter_iou=np.array(prob_filtered > 0.0, dtype='bool')
        boxe_filtered=boxe_filtered[filter_iou]
        prob_filtered=prob_filtered[filter_iou]
        classes_num_filtered=classes_num_filtered[filter_iou]

        result=[]
        for i in range(len(boxe_filtered)):
            result.append([self.para.CLASSES[classes_num_filtered[i]], boxe_filtered[i][0], boxe_filtered[i][1], boxe_filtered[i][2], boxe_filtered[i][3], prob_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb=min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
        lr=min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection=tb*lr
        return intersection/(box1[2]*box1[3]+box2[2]*box2[3]-intersection)

    def image_detector(self, image, wait=0):
        #image=cv2.imread(imname)

        result=self.detect(image)

        self.draw_result(image, result)
        image=cv2.resize(image,(600,600))
        return image
        #cv2.imshow('Image', image)
        #cv2.waitKey(wait)

para=Parameter
D=Detector(para)
>>>>>>> 42493b708c3f0dcec3ead7688f05a4adf07b7508
