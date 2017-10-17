# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:11:00 2017

@author: 颜
"""
import dictionary as dic
import tensorflow as tf
import numpy as np

#----------set parament---------------------------
loadfile="2.txt"                                    #文件
dim=100                                             #词向量的维数
vocabulary_size=50000                               #词向量的个数
batch_size=64                                       #一批样本的词数
sample_num=64                                       #每个样本词对应标记的个数(这包括了正标记和负采样得到的负标记)
sample_window=50000                                  #负采样的范围，只从前sample_window个词向量中负采样
context_window=6                                    #考虑前后context_window个词作为正标记的采样区
train_step_num=1000000                               #训练轮数
check_point=2000                                    #训练打印点
learnning_rate=0.01                                 #学习率
show_word_num=16                                    #打印前show_word_num个词
related_word_num=16                                 #打印的最近相关词数
b_s=batch_size*sample_num                           #每个batch的标记矩阵的词向量个数，

#----------build dictionary------------------------
a=dic.readfile(loadfile)
dictionary,reverse_dictionary,count,word_number=dic.init_dictionary(a,vocabulary_size)
#sample_probabilities=dic.sample_probability(count[0:sample_window])
sample_probabilities=None
#--------build graph-------------------------------
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

#reverse_labels_mark=1-labels_mark
#NCE_LOSS=tf.reduce_sum(labels_mark*tf.log(1+tf.exp(-output))+reverse_labels_mark*tf.log(1+tf.exp(output)))

NCE_LOSS=tf.reduce_sum(tf.maximum(output,0)-output*labels_mark+tf.log(1+tf.exp(-tf.abs(output))))
train_step = tf.train.GradientDescentOptimizer(learnning_rate).minimize(NCE_LOSS)

#----------Normalized final emmbeddings-----------------
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_examples = np.random.choice(100,show_word_num,replace=False)
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
similarity = tf.matmul(
    valid_embeddings, normalized_embeddings, transpose_b=True)

#------------------GGG!!!-------------------------------
sess.run(tf.global_variables_initializer())

pin=0
average_loss=0.
current_loss=0.
period=0

for i in range(1,train_step_num+1):
    pre_pin=pin
    batch_feed,labels_feed,labels_mark_feed,pin=dic.generate_batch(word_number,
                                                                   count,
                                                                   batch_size,
                                                                   pin,
                                                                   context_window,
                                                                   sample_num,
                                                                   sample_window,
                                                                   sample_probabilities)
    if pre_pin>pin:
        period+=1
    feed_dict={batch:batch_feed,labels:labels_feed,labels_mark:labels_mark_feed}
    _,current_loss=sess.run([train_step,NCE_LOSS],feed_dict=feed_dict)
    average_loss+=current_loss
    if i==1:
        print("LOSS after initialized",
              current_loss/b_s)
    if i%check_point==0:
        print("Training processing ",
              i*100/train_step_num,
              "%......AVERAGE LOSS:",
              average_loss/check_point/b_s)
        average_loss=0.
print("Process finished! Training period",
      period,
      "Final LOSS:",current_loss/b_s)
final_embeddings=normalized_embeddings.eval()

#---------plot imagine and save word_vector as np_file-----------
def plot_with_labels(low_dim_embs, labels, filename='111tsne.png'):
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

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    word = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, word)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")

np.save("vector.npy",final_embeddings)

#-------------------print some related word----------------------------
sim = similarity.eval()
for i in range(show_word_num):
    valid_word = reverse_dictionary[valid_examples[i]]
    nearest = (-sim[i, :]).argsort()[1:related_word_num + 1]
    log_str = 'Nearest to %s:' % valid_word
    for k in range(related_word_num):
        close_word = reverse_dictionary[nearest[k]]
        log_str = '%s %s,' % (log_str, close_word)
    print(log_str)



