# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:50:06 2017

@author: 颜
"""

import dictionary as dic
import tensorflow as tf
import numpy as np
import os
import tqdm as tqdm
import pickle

class embedding():
    def __init__(self,file=None,word_list=None,vocabulary_size=None):
        if not (file or word_list):
            print("A source file or word_list should be given!!!")
            os._exit()
        self.vocabulary_size=vocabulary_size
        if word_list:
            self.dictionary,self.reverse_dictionary,self.count,self.word_number=dic.init_dictionary(word_list,self.vocabulary_size)
        else:
            word_list=dic.readfile(file)
            self.dictionary,self.reverse_dictionary,self.count,self.word_number=dic.init_dictionary(word_list,self.vocabulary_size)
        self.parament=None
        self.embeddings=None

    def training(self,parament):
        if not len(parament)==8:
            print("Parament invalid!!!")
            os._exit()

        self.parament=parament
        dim=self.parament[0]
        batch_size=self.parament[1]
        sample_num=self.parament[2]
        sample_window=self.parament[3]
        context_window=self.parament[4]
        train_step_num=self.parament[5]
        check_point=self.parament[6]
        learnning_rate=self.parament[7]
        b_s=batch_size*sample_num
        vocabulary_size=self.vocabulary_size

        print('vector dimension=',dim,'\n',
              'batch size=',batch_size,'\n',
              'sample number=',sample_num,'\n',
              'sample window=',sample_window,'\n',
              'context window=',context_window,'\n',
              'training turns=',train_step_num,'\n',
              'check point=',check_point,'\n',
              'initial learning rate=',learnning_rate)
        sess = tf.InteractiveSession()
        batch=tf.placeholder(tf.int32,[batch_size])
        labels=tf.placeholder(tf.int32,[batch_size,sample_num])
        labels_mark=tf.placeholder(tf.float32,[batch_size,sample_num])
        embeddings=tf.Variable(tf.truncated_normal([vocabulary_size, dim],stddev=1.0,))
        weight=tf.Variable(tf.truncated_normal([vocabulary_size, dim],stddev=1.0,))
        bias=tf.Variable(tf.zeros([batch_size]))
        embedding_batch=tf.nn.embedding_lookup(embeddings,batch)
        embedding_weight=tf.nn.embedding_lookup(weight,labels)
        embedding_bias=tf.nn.embedding_lookup(bias,labels)
        output=tf.reduce_sum(tf.einsum("xz,xyz->xyz",embedding_batch,embedding_weight),2)+embedding_bias
        NCE_LOSS=tf.reduce_sum(tf.maximum(output,0)-output*labels_mark+tf.log(1+tf.exp(-tf.abs(output))))
        train_step=tf.train.GradientDescentOptimizer(learnning_rate).minimize(NCE_LOSS)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        sess.run(tf.global_variables_initializer())

        pin=0
        average_loss=0.
        current_loss=0.
        period=0
        self.t=tqdm.tqdmt=tqdm.tqdm(range(train_step_num),desc="Training",ncols=130,ascii=True,unit="Batch")
        info={'INIT_LOSS':0.,'LOSS':0.,'period':0.}
        for i in self.t:
            pre_pin=pin
            batch_feed,labels_feed,labels_mark_feed,pin=dic.generate_batch(self.word_number,
                                                                           self.count,
                                                                           batch_size,
                                                                           pin,
                                                                           context_window,
                                                                           sample_num,
                                                                           sample_window)
            if pre_pin>pin:
                period+=1
            feed_dict={batch:batch_feed,labels:labels_feed,labels_mark:labels_mark_feed}
            _,current_loss=sess.run([train_step,NCE_LOSS],feed_dict=feed_dict)
            average_loss+=current_loss
            if i==0:
                info['INIT_LOSS']=current_loss/b_s
                self.t.set_postfix(**info)
            if (i+1)%check_point==0:
                info['LOSS']=average_loss/check_point/b_s
                info['period']=period
                average_loss=0.
                self.t.set_postfix(**info)
        print("Training finished!!!")
        self.t.__del__()
        self.embeddings=normalized_embeddings.eval()

    def draw(self,filename='111tsne.png'):
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            tsne = TSNE(perplexity=30,
                        n_components=2,
                        init='pca',
                        n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(self.embeddings[:plot_only,:])
            labels = [self.reverse_dictionary[i] for i in range(plot_only)]
            assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
            plt.figure(figsize=(18, 18))
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i,:]
                plt.scatter(x, y)
                plt.annotate(label,
                             xy=(x, y),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')
            plt.savefig(filename)
        except ImportError:
            print("Please install sklearn and matplotlib to visualize embeddings.")

    def save(self,
             embed_fn="embedding.npy",
             dictionary_fn="dictionary.pkl"):
        np.save(embed_fn,self.embeddings)
        dictionary_info=[self.dictionary,self.reverse_dictionary,self.count,self.word_number]
        file=open(dictionary_fn,'wb')
        pickle.dump(dictionary_info,file)
        file.close()
        print("File saved as:\n",
              os.path.join(os.getcwd(),embed_fn),
              '\n',
              os.path.join(os.getcwd(),dictionary_fn))

    def load(self,
             embed_fn="embedding.npy",
             dictionary_fn="dictionary.pkl"):
        if not (os.path.splitext(embed_fn)[1]=='.npy' and os.path.splitext(dictionary_fn)[1]=='.pkl'):
            print("Ivalid File Name!!!")
            os._exit
        if not (os.path.exists(embed_fn) and os.path.exists(dictionary_fn)):
            print("No Such File!!!")
            os._exit()
        self.embeddings=np.load(embed_fn)
        dictionary_info=list()
        file=open(dictionary_fn,'rb')
        dictionary_info=pickle.load(file)
        self.dictionary=dictionary_info[0]
        self.reverse_dictionary=dictionary_info[1]
        self.count=dictionary_info[2]
        self.word_number=dictionary_info[3]
        print("File Found and Loaded!!!")

    def embed(self,word_list):
        if not self.embeddings.all():
            print("NO Embedding Invalid!!!")
            os._exit
        word_number=list()
        for word in word_list:
            if isinstance(word,str):
                word=word.encode()
            if word in self.dictionary.keys():
                word_number.append(self.dictionary[word])
            else:
                word_number.append(self.dictionary[b'UNK'])
        if len(word_number)>0:
            embedding_list=self.embeddings[word_number,:]
        return embedding_list