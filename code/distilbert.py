import torch
from torch import optim
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

def Max_Index(array):
	max_index = 0
	for i in range(len(array)):
		if(array[i]>array[max_index]):
			max_index = i
	return max_index

def Get_Report(true_labels, pred_labels, labels=None, digits=4):
	recall = metrics.recall_score(true_labels, pred_labels, average='macro', zero_division=0)
	precision = metrics.precision_score(true_labels, pred_labels, average='macro', zero_division=0)
	macrof1 = metrics.f1_score(true_labels, pred_labels, average='macro', zero_division=0)
	microf1 = metrics.f1_score(true_labels, pred_labels, average='micro', zero_division=0)
	acc = metrics.accuracy_score(true_labels, pred_labels)
	return recall, precision, macrof1, microf1, acc

# Try some other models
# Make DistilBert model within the class similar to the above models with 3 types of inputs
# We are given sentences which needs to be classified in one of the two classes and hence we need to make a distilBert class for this kind of model

class DistilBertModels(nn.Module):
	def __init__(self, model_name, num_labels):
		super(DistilBertModels, self).__init__()
		self.model = DistilBertModel.from_pretrained(model_name)
		self.linear = nn.Linear(768, num_labels)
		self.softmax = nn.Softmax(dim=1)
		self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

	def forward(self, input_ids, attention_mask):
		# ValueError: Input Tensor  is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.
		# Resolve this error by converting the input_ids to a tensor
		input_ids =  torch.tensor(input_ids)
		#input_ids = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=True)).unsqueeze(0)
		attention_mask = torch.tensor([1] * len(input_ids[0])).unsqueeze(0)
		_, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
		output = self.linear(pooled_output)
		output = self.softmax(output)
		return output
	
	def train(self, train_x, train_y, test_x, test_y):
		criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		optimizer = optim.Adam(self.parameters(), lr=0.00001)
		v_test_x = torch.autograd.Variable(torch.Tensor(np.array([np.array(obj) for obj in test_x])))
		
		df = pd.DataFrame()
		
		for epoch in range(100):

			optimizer.zero_grad()

			rand_index = np.random.choice(len(train_x), size=32, replace=False)

			batch_x = torch.autograd.Variable(torch.LongTensor(np.array([np.array(obj) for i, obj in enumerate(train_x) if i in rand_index])))
			batch_y = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(train_y) if i in rand_index])))

			train_prediction = self.forward(batch_x, batch_y)

			loss = criterion(train_prediction, batch_y)

			loss.backward()

			optimizer.step()

			prediction_test = self.forward(v_test_x, v_test_y)
			pre_labels = [Max_Index(line) for line in prediction_test.data.numpy()]
			recall, precision, macrof1, microf1, acc = Get_Report(test_y, pre_labels)
			df = pd.concat([df,pd.DataFrame({'recall':[recall],'precision':[precision],'macrof1':[macrof1],'microf1':[microf1],'acc':[acc]})],axis=0,ignore_index=True)
			print("[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}    accuracy:{:.4%}".format(epoch, recall, precision, macrof1, microf1, acc))
		
		df.to_csv("../metrics_dist.csv")
