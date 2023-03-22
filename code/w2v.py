import gensim.models.word2vec
import numpy as np
from gensim.models.word2vec import Word2Vec
from coarse2fine import C2F
import os
import gensim.downloader as api
import tensorflow as tf
import torch

#dir = os.getcwd()
#dir = dir.replace("\code", "")
#vocabulary = set(open(dir + "/data/text8.txt").read().split(" "))
#corpus = gensim.models.word2vec.Text8Corpus(dir + "/data/text8.txt")
#model = Word2Vec(corpus, min_count=1)
#model.save("word2vec_model")
print(tf.config.list_physical_devices("GPU"))
print(torch.cuda.is_available())