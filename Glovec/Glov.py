# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:13:39 2017

@author: 颜
"""
import tensorflow as tf
import Global_statistic as Gs
import numpy as np
#----------set parament---------------------------
loadfile="2.txt"                                    #文件
dim=100                                             #词向量的维数
matrix_size=50000                                   #词向量的个数
context_window=6                                    #上下文窗口
batch_size=1000                                      #批次大小
x_max=100.0
a=0.75                                              #两个超参
show_word_num=10                                    #展示单词数量
related_word_num=10                                 #近义词数量
train_step_num=1000000                               #训练轮数
check_point=1000                                    #训练打印点
learnning_rate=0.01                                 #学习率


#---------build matrix-----------------------------
file=Gs.readfile(loadfile)
dictionary,reverse_dictionary,count,word_number=Gs.init_dictionary(file,matrix_size)
list_X=Gs.generate_list(matrix_size,word_number,context_window)

#--------build graph--------------------------------
sess = tf.InteractiveSession()

word_i=tf.placeholder(dtype=tf.int32,shape=[batch_size])
word_j=tf.placeholder(dtype=tf.int32,shape=[batch_size])
X_ij=tf.placeholder(dtype=tf.float32,shape=[batch_size])

embeddings_i=tf.Variable(tf.truncated_normal([matrix_size, dim],stddev=1.0,))
embeddings_j=tf.Variable(tf.truncated_normal([matrix_size, dim],stddev=1.0,))
bias_i=tf.Variable(tf.zeros([matrix_size]))
bias_j=tf.Variable(tf.zeros([matrix_size]))

w_i=tf.nn.embedding_lookup(embeddings_i,word_i)
w_j=tf.nn.embedding_lookup(embeddings_j,word_j)
b_i=tf.nn.embedding_lookup(bias_i,word_i)
b_j=tf.nn.embedding_lookup(bias_j,word_j)
f_ij=tf.minimum(1.0,tf.pow(X_ij/x_max,a))

wiwj=tf.reduce_sum(tf.einsum("xy,xy->xy",w_i,w_j),1)
LOSS=tf.reduce_sum(f_ij*tf.abs((wiwj+b_i+b_j-tf.log(X_ij))))
#LOSS=tf.reduce_sum(f_ij*tf.pow((wiwj+b_i+b_j-tf.log(X_ij)),2))

train_step = tf.train.GradientDescentOptimizer(learnning_rate).minimize(LOSS)

#----------Normalized final emmbeddings-----------------
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings_i), 1, keep_dims=True))
normalized_embeddings = embeddings_i / norm
valid_examples = np.random.choice(100,show_word_num,replace=False)
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
similarity = tf.matmul(
    valid_embeddings, normalized_embeddings, transpose_b=True)

#------------------GGG!!!-------------------------------
sess.run(tf.global_variables_initializer())

average_loss=0.

for i in range(1,train_step_num+1):
    word_i_feed,word_j_feed,X_ij_feed=Gs.generate_batch(list_X,batch_size)
    feed_dict={word_i:word_i_feed,word_j:word_j_feed,X_ij:X_ij_feed}
    _,current_loss=sess.run([train_step,LOSS],feed_dict=feed_dict)
    average_loss+=current_loss
    if i%check_point==0:
                print("Training processing ",
                      i*100/train_step_num,
                      "%......AVERAGE LOSS:",
                      average_loss/check_point/batch_size)
                average_loss=0.
print("Process finished!!!",
      "Final LOSS:",current_loss/batch_size)
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



