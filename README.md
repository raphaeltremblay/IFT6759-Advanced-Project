
## Overview
- code/ 
  This directory contains the source code of our approach.
- data/ 
  This directory contains two datasets used for evaluation.


## Datasets
The dataset named as X-Y denotes a text classification task Y on data source X. For two sentence-level tasks -- SC (sentence classification) and SSR (sentence semantics recognition), an example <x,y> in each line denotes a sentence x and its label y. each of them can be solely used to evaluate single-sentence classification tasks. For the word-level task -- SRL (semantic role labeling), an example <x,i,y> in each line denotes a sentence x, a subordinate word index i and a corresponding label y. It can be further used to evaluate sequential-text classification tasks. We will keep updating them to provide more reliable version(s), including correcting wrongly-annotated labels and adding more training/testing examples. The up-to-date version can be directly downloaded from this repository. In summary:
* COR for cooking recipes and MAM for maintenance manuals.
* COR-SC.txt  is the dataset for sentence-level classiﬁcation (ST1) to identify whether a sentence is describing an action or a statement. The format is <Sentence, Label>.
* COR-SSR.txt is the dataset for sentence-level semantics recognition (ST2) to recognize the semantics of a Statement sentence to control the execution of following actions. The format is <Sentence, Label, Y/N> in which Y denotes the sentence belongs to the label while N not (so, you could ignore all examples with N notations, just focus on the examples with Y notations).
* COR-SRL.txt is the dataset for word-level semantic role labeling to assign semantic roles to words in an Action sentence. The format is <Word, Word Postion in the Sentence, Label>.

## How to run the code
To run the experiments, we recommend using the advproj.ipynb notebook on Colab or any other software that reads Jupyter Notebook, and running every cell inside it. 
* The first cell will prompt a request to connect to your Google Account in order to mount the notebook to it.
* The second cell will download the directory from this Github. It is possible to choose which branch will be downloaded.
* The last cell runs the actual experiment. It has the format "!python3 main.py <model_name> <dataset>". For model_name, one can choose between "word2vec_model", "bert_pretrained" or "distilbert_pretrained" and the datasets are "COR" or "MAM".
