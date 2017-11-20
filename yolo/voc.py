# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:21:41 2017

@author: 颜
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config
from tqdm import tqdm

class pascal_voc():
    def __init__(self,para):
        self.devkil_path=os.path.join(para.VOC_PATH,'VOCdevkit')
        self.data_path=os.path.join(self.devkil_path,para.VOC_VERSION)
        self.cache_path=para.CACHE_PATH
        self.batch_size=para.batch_size
        self.image_size=para.image_size
        self.cell_size=para.S
        self.label_size=5+para.label_nums
        self.classes=para.CLASSES
        self.class_to_ind=dict(zip(self.classes, range(len(self.classes))))
        self.flipped=para.FLIPPED
        self.rebuild=para.REBUILD
        self.phase=para.PHASE
        self.cursor=0
        self.epoch=1
        self.length=0
        self.gt_labels=None
        self.prepare()

    def generate_batch(self):
        images=np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels=np.zeros((self.batch_size, self.cell_size, self.cell_size, self.label_size))
        count = 0
        while count<self.batch_size:
            imname=self.gt_labels[self.cursor]['imname']
            flipped=self.gt_labels[self.cursor]['flipped']
            images[count,:,:,:]=self.image_read(imname,flipped)
            labels[count,:,:,:]=self.gt_labels[self.cursor]['label']
            count+=1
            self.cursor+=1
            if self.cursor>=self.length:
                np.random.shuffle(self.gt_labels)
                self.cursor=0
                self.epoch+=1
        return images,labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image,(self.image_size,self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:,::-1,:]
        return image

    def prepare(self):
        gt_labels=self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped']=True
                gt_labels_cp[idx]['label']=gt_labels_cp[idx]['label'][:,::-1,:]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i,j,0]==1:
                            gt_labels_cp[idx]['label'][i,j,1]=self.image_size-1-gt_labels_cp[idx]['label'][i,j,1]
            gt_labels+=gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels=gt_labels
        self.length=len(gt_labels)
        print(self.length,"images in total")
        return gt_labels

    def load_labels(self):
        cache_file=os.path.join(self.cache_path,'pascal_'+self.phase+'_gt_labels.pkl')
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading image_labels from:'+cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Generating image_labels from:'+self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase=='train':
            txtname=os.path.join(self.data_path,'ImageSets','Main','trainval.txt')
        else:
            txtname=os.path.join(self.data_path,'ImageSets','Main','val.txt')
        with open(txtname,'r') as f:
            self.image_index=[x.strip() for x in f.readlines()]

        gt_labels=[]
#        for index in self.image_index:
#            label,num=self.load_pascal_annotation(index)
        for i in tqdm(range(len(self.image_index)),desc="Loading",ncols=110,ascii=True,unit="images"):
            label,num=self.load_pascal_annotation(self.image_index[i])
            if num==0:
                continue
            imname=os.path.join(self.data_path,'JPEGImages',self.image_index[i]+'.jpg')
            gt_labels.append({'imname':imname,'label':label,'flipped':False})
        print('Saving image_labels to: '+cache_file)
        with open(cache_file,'wb') as f:
            pickle.dump(gt_labels,f)
        return gt_labels

    def load_pascal_annotation(self, index):
        imname=os.path.join(self.data_path,'JPEGImages',index +'.jpg')
        im=cv2.imread(imname)
        h_ratio=1.0*self.image_size/im.shape[0]
        w_ratio=1.0*self.image_size/im.shape[1]
        label=np.zeros((self.cell_size,self.cell_size,self.label_size))
        filename=os.path.join(self.data_path,'Annotations',index +'.xml')
        tree=ET.parse(filename)
        objs=tree.findall('object')
        for obj in objs:
            bbox=obj.find('bndbox')
            x1=max(min((float(bbox.find('xmin').text)-1)*w_ratio,self.image_size-1),0)
            y1=max(min((float(bbox.find('ymin').text)-1)*h_ratio,self.image_size-1),0)
            x2=max(min((float(bbox.find('xmax').text)-1)*w_ratio,self.image_size-1),0)
            y2=max(min((float(bbox.find('ymax').text)-1)*h_ratio,self.image_size-1),0)
            cls_ind=self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes=[(x2+x1)/2.0,(y2+y1)/2.0,x2-x1,y2-y1]
            x_ind=int(boxes[0]*self.cell_size/self.image_size)
            y_ind=int(boxes[1]*self.cell_size/self.image_size)
            if label[y_ind,x_ind,0]==1:
                continue
            label[y_ind,x_ind,0]=1
            label[y_ind,x_ind,1:5]=boxes
            label[y_ind,x_ind,5+cls_ind]=1
        return label,len(objs)