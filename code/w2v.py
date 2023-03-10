import gensim.models.word2vec
import numpy as np
from gensim.models.word2vec import Word2Vec
from coarse2fine import C2F
import os
import gensim.downloader as api


dir = os.getcwd()
dir = dir.replace("\code", "")
#corpus = api.load('text8')
vocabulary = set(open(dir + "/data/text8.txt").read().split(" "))
corpus = gensim.models.word2vec.Text8Corpus(dir + "/data/text8.txt")
model = Word2Vec(corpus, min_count=1)
word= "questionnaires"
print((word in vocabulary))
if(word in vocabulary):
    print(model.wv[word])
#model.save("word2vec")
#print(model.wv.most_similar('hi'))
#print(model.wv["wi"])