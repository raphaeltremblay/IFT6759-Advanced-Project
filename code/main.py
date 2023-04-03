import gensim.models.word2vec
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim import downloader
from coarse2fine import C2F
import os
import sys
#from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
from transformers import AutoModel, AutoTokenizer

all_SC, all_SSR, all_SRL = [], [], []
label_SC, label_SSR, label_SRL = set(), set(), set()

dir = ".."
#Choose which dataset to use below between "COR" and "MAM"
dataset = sys.argv[2]
#Choose which embedding model to use below between "word2vec_model", "bert_pretrained"
model_name = sys.argv[1]


w2v_embdding_size = 100

if model_name=="word2vec_model":
	model = Word2Vec.load(model_name)

if model_name=="bert_pretrained":
	model = AutoModel.from_pretrained('bert-base-uncased')
	model.to("cuda")

for line in open(dir+"/data/"+dataset+"-SC.txt").read().split("\n"):
	objs = line.lower().split(", ")
	if len(objs)==2:
		all_SC.append(objs)
		label_SC.add(objs[-1])

for line in open(dir+"/data/"+dataset+"-SSR.txt").read().split("\n"):
	objs = line.lower().split(", ")
	if line.endswith(", y") and len(objs)==3:
		objs = objs[:-1]
		all_SSR.append(objs)
		label_SSR.add(objs[-1])

for line in open(dir+"/data/"+dataset+"-SRL.txt").read().split("\n"):
	objs = line.lower().split(", ")
	if len(objs)==3:
		all_SRL.append(objs)
		label_SRL.add(objs[-1])

print(len(all_SC))
print(all_SC[0:10])
print(label_SC)

print(len(all_SSR))
print(all_SSR[0:10])
print(label_SSR)

print(len(all_SRL))
print(all_SRL[0:10])
print(label_SRL)



ratio = 0.80
train_SC,  test_SC  = all_SC[:int(len(all_SC)*ratio)],   all_SC[int(len(all_SC)*ratio):]
train_SSR, test_SSR = all_SSR[:int(len(all_SSR)*ratio)], all_SSR[int(len(all_SSR)*ratio):]
train_SRL, test_SRL = all_SRL[:int(len(all_SRL)*ratio)], all_SRL[int(len(all_SRL)*ratio):]
print(len(train_SC), len(test_SC))
print(len(train_SSR), len(test_SSR))
print(len(train_SRL), len(test_SRL))



vocabulary = set(open(dir + "/data/text8.txt").read().split(" "))

label_SC  = list(label_SC)
label_SSR = list(label_SSR)
label_SRL = list(label_SRL)


def Encode_Sentence_Data(array, label_map):
	embeddings, labels = [], []
	for line in array:
		words = line[0].split(" ")
		sentence = line[0]
		label = line[1]

		mat = []
		if model_name=="word2vec_model":
			for word in words:
				if(word in vocabulary):
					mat.append(model.wv[word])
				else:
					mat.append(model.wv["a"])
			while len(mat)<10:
				mat.append(model.wv["a"])
			mat = mat[:10]
			embeddings.append(mat)

		if model_name=="bert_pretrained":
			tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
			tokenized = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
			tokenized = {k: torch.tensor(v).to("cuda") for k, v in tokenized.items()}
			hidden = model(**tokenized)
			cls = hidden.last_hidden_state[:, 0, :]
			embeddings.append(cls)

		labels.append(label_map.index(label))

	print("Encoding Sentence Finished Once")
	return embeddings, labels

def Encode_Word_Data(array, label_map):
	embeddings, wembeddings, labels = [], [], []
	for line in array:
		words = line[0].split(" ")
		label = line[-1]

		mat = []
		if model_name == "word2vec_model":
			for word in words:
				if(word in vocabulary):
					mat.append(model.wv[word])
				else:
					mat.append(model.wv["a"])
			while len(mat)<10:
				mat.append(model.wv["a"])
			mat = mat[:10]

			embeddings.append(mat)

			index = int(line[1])
			center_word = line[0].split(" ")[index]
			if (center_word in vocabulary):
				rep = list(np.array(model.wv[center_word]))
				rep.extend([index*1.0])
				rep = [float(obj) for obj in rep]
				wembeddings.append(rep)
			else:
				rep = list(np.array(model.wv["a"]))
				rep.extend([index * 1.0])
				rep = [float(obj) for obj in rep]
				wembeddings.append(rep)

		if model_name=="bert_pretrained":
			tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
			tokenized = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
			tokenized = {k: torch.tensor(v).to("cuda") for k, v in tokenized.items()}
			hidden = model(**tokenized)
			cls = hidden.last_hidden_state[:, 0, :]
			embeddings.append(cls)
			index = int(line[1])
			center_word = line[0].split(" ")[index]
			word_embedding = tokenizer(center_word, padding=True, truncation=True, return_tensors="pt")
			word_embedding = {k: torch.tensor(v).to("cuda") for k, v in word_embedding.items()}
			hidden_word = model(**word_embedding)
			cls_word = hidden_word.last_hidden_state[:,0,:]
			wembeddings.append(cls_word)
		labels.append(label_map.index(label))

		# print(line)
	print("Encoding Words Finished Once")
	return embeddings, wembeddings, labels

train_x1, train_y1 = Encode_Sentence_Data(train_SC, label_SC)
test_x1,  test_y1  = Encode_Sentence_Data(test_SC, label_SC)

train_x2, train_y2 = Encode_Sentence_Data(train_SSR, label_SSR)
test_x2,  test_y2  = Encode_Sentence_Data(test_SSR, label_SSR)

train_x3s, train_x3w, train_y3 = Encode_Word_Data(train_SRL, label_SRL)
test_x3s,  test_x3w,  test_y3  = Encode_Word_Data(test_SRL, label_SRL)


if model_name=="word2vec_model" or model_name=="bert_pretrained":
	c2f = C2F(len(label_SC), len(label_SSR), len(label_SRL))
	c2f.train(train_x1, train_y1, test_x1,  test_y1, train_x2, train_y2, test_x2,  test_y2, train_x3s, train_x3w, train_y3, test_x3s,  test_x3w,  test_y3)

if model_name=="distilbert-base-uncased":
	distil = DistilBertModels(model_name="distilbert-base-uncased", num_labels=len(label_SC))
	distil.train(train_x1, train_y1, test_x1,  test_y1)