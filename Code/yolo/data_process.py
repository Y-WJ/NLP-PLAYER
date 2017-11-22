<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:56:34 2017

@author: 颜
"""
import cv2
from config import Parameter
import os
import re
import pickle
import numpy as np
import tqdm

number=re.compile('^\d{1,4}$')
isfile=re.compile('^\d*.jpg$')

class crowdai():
    def __init__(self,para):
        self.para=para
        self.dic={'Car':6,'Truck':5,'Pedestrian':14}
        self.rev_dic={6:'car',5:'truck',14:'pedestrian'}
        self.samples=self.load_image()
        self.len=len(self.samples)
        self.pin=0
        self.epoch=0

    def load_image(self):
        if os.path.isfile('cache_samples.pickle'):
            print('LOADING FROM CACHE!!!')
            with open('cache/crowdai_cache_samples.pickle','rb') as f:
                samples=pickle.load(f)
        else:
            samples=[]
            f=open(os.path.join(self.para.DATA_PATH,'labels.csv'),'r')
            k=f.readlines()
            l=len(k)
            image=[]
            filename='none'
            for i in tqdm.tqdm(range(l),desc="Loading",ncols=120,ascii=True,unit="labels"):
                line=k[i].split(',')
                if number.match(line[0]) and number.match(line[1]) and number.match(line[2]) and number.match(line[3]) and isfile.match(line[4]) and (line[5]=='Car' or line[5]=='Pedestrian' or line[5]=='Truck'):
                    if line[4]!=filename:
                        if len(image)>1:
                            samples.append(image)
                        image=[line[4],np.zeros((7,7,25),dtype=float)]
                        filename=line[4]
                    image[1]=self.write_to_label(image[1],line)
            np.random.shuffle(samples)
            with open('cache/crowdai_cache_samples.pickle','wb') as f:
                pickle.dump(samples,f)
            print("FILE SAVED AS PICKLE FILES!!!")
        return samples

    def readimage(self,image):
        image=cv2.imread(os.path.join(self.para.DATA_PATH,image))
        image=cv2.resize(image,(self.para.image_size,self.para.image_size))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image=(image/255.0)*2.0-1.0
        return image

    def generate_batch(self):
        image=np.zeros((self.para.batch_size, self.para.image_size, self.para.image_size, 3))
        label=np.zeros((self.para.batch_size,7,7,25),dtype=float)
        count = 0
        while count<self.para.batch_size:
            image[count,:,:,:]=self.readimage(self.samples[self.pin][0]).astype(np.float)
            label[count,:,:,:]=self.samples[self.pin][1]
            count+=1
            self.pin+=1
            if self.pin>=self.len:
                np.random.shuffle(self.samples)
                self.pin=0
                self.epoch+=1
        return image,label

    def write_to_label(self,label,line):
        h_ratio=1.0*self.para.image_size/1920
        w_ratio=1.0*self.para.image_size/1200
        x1=max(min((float(line[0])-1)*w_ratio,self.para.image_size-1),0)
        y1=max(min((float(line[2])-1)*h_ratio,self.para.image_size-1),0)
        x2=max(min((float(line[1])-1)*w_ratio,self.para.image_size-1),0)
        y2=max(min((float(line[3])-1)*h_ratio,self.para.image_size-1),0)
        cls_ind=self.dic[line[5]]
        boxes=[(x2+x1)/2.0,(y2+y1)/2.0,x2-x1,y2-y1]
        x_ind=int(boxes[0]*7/self.para.image_size)
        y_ind=int(boxes[1]*7/self.para.image_size)
        label[y_ind,x_ind,0]=1
        label[y_ind,x_ind,1:5]=boxes
        label[y_ind,x_ind,5+cls_ind]=1
        return label
