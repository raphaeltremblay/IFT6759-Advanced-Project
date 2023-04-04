import os
import sys
import torch
import torch_xla
import numpy as np
from coarse2fine import C2F
from gensim import downloader
import gensim.models.word2vec
from gensim.models import Word2Vec
import torch_xla.core.xla_model as xm
from transformers import AutoModel, AutoTokenizer


all_SC, all_SSR, all_SRL = [], [], []
label_SC, label_SSR, label_SRL = set(), set(), set()

# dir = os.getcwd()
# dir = dir.replace("\code", "")
dir = ".."
#Choose which dataset to use below between "COR" and "MAM"
dataset = sys.argv[2]
#Choose which embedding model to use below between "word2vec_model" and "bert_pretrained",
model_name = sys.argv[1]

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



w2v_embdding_size = 100

if model_name=="word2vec_model":
	model = Word2Vec.load(model_name)
	
if model_name=="bert_pretrained":
	model = AutoModel.from_pretrained('bert-base-uncased')
# 	dev = xm.xla_device()
# 	model = model.to(dev)
	model.to("cuda")

if model_name=="distilbert_pretrained":
	model = AutoModel.from_pretrained('distilbert-base-uncased')
# 	dev = xm.xla_device()
# 	model = model.to(dev)
	model.to("cuda")
	
	
vocabulary = set(open(dir + "/data/text8.txt").read().split(" "))

label_SC  = list(label_SC)
label_SSR = list(label_SSR)
label_SRL = list(label_SRL)

def Encode_Sentence_Data(array, label_map):
	embeddings, labels = [], []
	mat = []
	if model_name == "word2vec_model":
		for line in array:
			words = line[0].split(" ")
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
		sentences_list = [i[0] for i in array]
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		tokenized = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
		tokenized = {k: v.clone().detach().to("cuda") for k, v in tokenized.items()}
		with torch.no_grad():
			hidden = model(**tokenized)
		cls = hidden.last_hidden_state[:, 0, :]
		embeddings = cls.tolist()
		
	if model_name=="distilbert_pretrained":
		sentences_list = [i[0] for i in array]
		tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
		tokenized = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
		tokenized = {k: v.clone().detach().to("cuda") for k, v in tokenized.items()}
		with torch.no_grad():
			hidden = model(**tokenized)
		cls = hidden.last_hidden_state[:, 0, :]
		embeddings = cls.tolist()

	for line in array:
		label = line[1]
		labels.append(label_map.index(label))

	print("Encoding Sentence Finished Once")
	return embeddings, labels

def Encode_Word_Data(array, label_map):
	embeddings, wembeddings, labels, center_words = [], [], [], []
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
			sentences_list = [i[0] for i in array]
			tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
			tokenized = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
			tokenized = {k: v.clone().detach().to("cuda") for k, v in tokenized.items()}
			with torch.no_grad():
				hidden = model(**tokenized)
			cls = hidden.last_hidden_state[:, 0, :]
			embeddings = cls.tolist()
			for line in array:
				index = int(line[1])
				center_word = line[0].split(" ")[index]
				center_words.append(center_word)
			word_embedding = tokenizer(center_word, padding=True, truncation=True, return_tensors="pt")
			word_embedding = {k: v.clone().detach().to("cuda") for k, v in word_embedding.items()}
			with torch.no_grad():
				hidden_word = model(**word_embedding)
			cls_word = hidden_word.last_hidden_state[:,0,:]
			wembeddings = cls_word.tolist()
		
		if model_name=="distilbert_pretrained":
			sentences_list = [i[0] for i in array]
			tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
			tokenized = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
			tokenized = {k: v.clone().detach().to("cuda") for k, v in tokenized.items()}
			with torch.no_grad():
				hidden = model(**tokenized)
			cls = hidden.last_hidden_state[:, 0, :]
			embeddings = cls.tolist()
			for line in array:
				index = int(line[1])
				center_word = line[0].split(" ")[index]
				center_words.append(center_word)
			word_embedding = tokenizer(center_word, padding=True, truncation=True, return_tensors="pt")
			word_embedding = {k: v.clone().detach().to("cuda") for k, v in word_embedding.items()}
			with torch.no_grad():
				hidden_word = model(**word_embedding)
			cls_word = hidden_word.last_hidden_state[:,0,:]
			wembeddings = cls_word.tolist()

	for line in array:
		label = line[-1]
		labels.append(label_map.index(label))

		# print(line)
	print("encoding finished")
	return embeddings, wembeddings, labels

train_x1, train_y1 = Encode_Sentence_Data(train_SC, label_SC)
test_x1,  test_y1  = Encode_Sentence_Data(test_SC, label_SC)

train_x2, train_y2 = Encode_Sentence_Data(train_SSR, label_SSR)
test_x2,  test_y2  = Encode_Sentence_Data(test_SSR, label_SSR)

train_x3s, train_x3w, train_y3 = Encode_Word_Data(train_SRL, label_SRL)
test_x3s,  test_x3w,  test_y3  = Encode_Word_Data(test_SRL, label_SRL)

# if model_name=="word2vec_model" or model_name=="bert_pretrained":
c2f = C2F(len(label_SC), len(label_SSR), len(label_SRL))
c2f.train(train_x1, train_y1, test_x1,  test_y1, train_x2, train_y2, test_x2,  test_y2, train_x3s, train_x3w, train_y3, test_x3s,  test_x3w,  test_y3)

# if model_name=="distilbert-base-uncased":
# 	distil = DistilBertModels(model_name="distilbert-base-uncased", num_labels=len(label_SC))
# 	distil.train(train_x1, train_y1, test_x1,  test_y1)
