# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:09:44 2017

@author: 颜
"""


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

model=Word2Vec(LineSentence("wiki_corpus.txt",limit=True),size=300,workers=multiprocessing.cpu_count())
model.save("wordvector")
word_vec=model.wv

#from gensim.models import Word2Vec
#import wordtovector as w
'''
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    wiki=WikiCorpus("enwiki-latest-pages-articles.xml.bz2",lemmatize=False)
    print("inited!!!")
'''

from gensim.corpora import WikiCorpus
import sys
import logging
import os.path
import multiprocessing


def write(text,dic,word_id,file):
    l=len(text)
    for i in range(l):
        if not text[i] in word_id:
            text[i]='unk'
        elif not word_id[text[i]] in dic:
            text[i]='unk'
    text=' '.join(text)+'\n'
    file.write(text)

if __name__=='__main__':
    __spec__=None
    wiki=WikiCorpus.load("wiki_corpus")
    pool=multiprocessing.Pool(multiprocessing.cpu_count())
    dic=wiki.dictionary.dfs
    word_id=wiki.dictionary.token2id
    discard=list()
    for key in dic:
        if dic[key]<=5:
            discard.append(key)
    for key in discard:
        dic.pop(key)
    file=open("wiki_corpus.txt",'a',encoding='utf-8')
    i=0.
    print("PROCESSING!!!")
    for text in wiki.get_texts():
        write(text,dic,word_id,file)
        i+=1
        if i%1000==0:
            print("processing",i/37500,"%")
    file.close()
'''





