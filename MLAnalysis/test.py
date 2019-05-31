#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################

import codecs
from numpy import asarray
from numpy import argmax
from numpy import zeros
from numpy import random
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import os
import time
import re

from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras import models

from keras.callbacks import TensorBoard
######################################
embedding_dims = 50 # Here 50/100/200/300
epochs = 2
k_cross_val=4
average="macro" # binary | micro | macro | weighted | samples
# class_weight={
# 	0 : 15.,
# 	1 : 50.,
# 	2 : 15.,
# 	3 : 10.}
skf=StratifiedKFold(n_splits=k_cross_val,random_state=25)
result_output = "ResultLSTM"+str(embedding_dims)+"d.csv"
embedding_file = 'glove.6B.'+str(embedding_dims)+'d.txt'
vocab_size=500
completeCitation="CompleteCitation"
file_dataset="SAR14.txt"
# Lemmatizer & Stemmer
lemmatizer=WordNetLemmatizer()
stemmer=SnowballStemmer('english',ignore_stopwords=True)
target_names=[
	"Negative",
	"Positive"]
################################################    Functions     #################################################
#
def lemma_word(word):
	"""This function take as args word and return its lemma
	
	Args : 
		- word : (str) a word that could lemmatize by the WordNetLemmatizer from nltk.stem
	
	Return : 
		- word : (str) a lemma of the word gives in args
	"""
	return lemmatizer.lemmatize(word)
#
def lemma_tokenizer(doc):
	""" This function take as args a doc that could be lemmatize.

	Args : 
		- doc : (str) a string that can be tokenize by the word_tokenize of nltk library
	
	Return : 
		- tokens : (list) a list of tokens where each token corresponds to a lemmatized word 
	"""
	tokens=word_tokenize(doc)
	tokens=[lemma_word(t) for t in tokens]
	return tokens
#
def stem_word(word):
	"""This function take as args word and return its stem
	
	Args : 
		- word : (str) a word that could be stemmed by the SnowballStemmer from nltk.stem.snowball
	
	Return : 
		- word : (str) a stem of the word gives in args
	"""
	return stemmer.stem(word)
#
def stem_tokenizer(doc):
	""" This function take as args a doc that could be stemmed.

	Args : 
		- doc : (str) a string that can be tokenize by the word_tokenize of nltk library
	
	Return : 
		- tokens : (list) a list of tokens where each token corresponds to a stemmed word 
	"""
	tokens=word_tokenize(doc)
	tokens=[stem_word(t) for t in tokens]
	return tokens
#
def tokenizer(doc):
	""" This function take as args a doc that could be tokenize.

	Args :
		- doc : (str) a string that can be tokenize by the word_tokenize of nltk library
	
	Return : 
		- tokens : (list) a list of tokens where each token corresponds to a word
	"""
	tokens=word_tokenize(doc)
	return tokens
#
###################################################    Main     ###################################################
#
dataset=codecs.open(
	filename=file_dataset,
	mode="r",
	encoding="utf-8")
citation=[]
label=[]
for line in dataset.readlines():
	line=line.split('"')
	citation.append(line[1])
	label.append(int(line[2].split('\n')[0].split(',')[1]))
tmp=[]
for value in label:
	if value<5:
		tmp.append(0)
	else:
		tmp.append(1)
label=tmp
tmp=None
TMP=[citation[0:10000],label[0:10000]]
dataset.close()
dataset=TMP
TMP=None
data=DataFrame.from_items([
	(completeCitation,dataset[0]),
	('Categories_num',dataset[1])])
dataset=None
citation=None
label=None
output_file=codecs.open(result_output,'w',encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tCross-score("+str(k_cross_val)+")\tLoss\tTime\tApproach\n")
lemma_citation=[]
stem_citation=[]
for citation in data[completeCitation]:
	lemma_citation.append(" ".join(lemma_tokenizer(citation)))
	stem_citation.append(" ".join(stem_tokenizer(citation)))
data["lemma_citation"]=lemma_citation
data["stem_citation"]=stem_citation
approaches=[data[completeCitation],data["lemma_citation"],data["stem_citation"]]
for approach in approaches:	
	tokenizer=Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(approach)
	tmp=tokenizer.texts_to_sequences(approach)
	word_index=tokenizer.word_index
	max_len=len(max(tmp,key=len))
	tmp=DataFrame(pad_sequences(# complete a sequence of tokenized words to match the length of the longest sentence 
		sequences=tmp,
		maxlen=max_len,
		padding='post'))
	X=tmp
	y=data.Categories_num
	tmp=None
	# data=concat([data[featuresList],tmp],axis=1)
	# tmp=None
	# X=data.drop(['Categories_num'],axis=1)
	# y=data.Categories_num
	accuracy_list=[]
	start=time.time()
	control=0
	for train_index,test_index in skf.split(X,y):
		NAME="LSTM-"+str(embedding_dims)+"D-epochs"+str(epochs)+"-"+str(approach.name)+str(control)+"-{}".format(int(time.time()))
		tensorboard=TensorBoard(log_dir='./logsLSTM/{}'.format(NAME))	
		X_train,X_test=[X.ix[train_index],X.ix[test_index]] 
		y_train,y_test=[y.ix[train_index],y.ix[test_index]]
		X_train=[X_train.iloc[:,3:],X_train.iloc[:,:3]] #seq_features,other_features
		X_test=[X_test.iloc[:,3:],X_test.iloc[:,:3]] #seq_features,other_features
		embeddings_index={}
		f=codecs.open(embedding_file,'r',encoding='utf-8')
		for line in f:# load pre-trained embedding
			values=line.split()
			word=values[0]
			coefs=asarray(values[1:],dtype='float32')
			embeddings_index[word]=coefs
		f.close()
		not_in_embedding=0
		embedding_matrix=random.uniform(-0.5,0.5,(len(word_index)+1,embedding_dims))
		for word,i in word_index.items():# building matrix
			embedding_vector=embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i]=embedding_vector
			else:
				not_in_embedding+=1
		print(not_in_embedding,"/",len(word_index)+1)
		##################  MODEL  ########################
		input_layer=layers.Input(
			shape=(X_train[0].shape[1],))
		embedding=layers.Embedding(
			input_dim=len(word_index)+1,
			output_dim=embedding_dims,
			weights=[embedding_matrix],
			input_length=X_train[0].shape[1],
			trainable=True,
			mask_zero=True)(input_layer)
		seq_features=layers.LSTM(
			units=embedding_dims,
			activation='tanh',
			go_backwards=True)(embedding)
		model=layers.Dense(
			units=128,
			activation='relu')(seq_features)
		# model=layers.Dropout(
		# 	rate = .4)(model)
		model=layers.Dense(
			units=len(target_names),
			activation='softmax')(model)
		model=models.Model(
			inputs=[input_layer],
			outputs=model)
		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=['accuracy'])
		model.fit(
			X_train[0],
			y_train,
			epochs=epochs,
			batch_size=20,
			# class_weight=class_weight,
			validation_data=(X_test[0],y_test),
			callbacks=[tensorboard])
		##################  MODEL  ########################
		val_loss,val_acc=model.evaluate(X_test[0],y_test)
		result=model.predict(X_test[0])
		y_pred=[]
		for sample in result:
			y_pred.append(argmax(sample))
		f1_score=round(metrics.f1_score(y_test,y_pred,average=average)*100,3)
		precision=round(metrics.precision_score(y_test,y_pred,average=average)*100,3)
		recall=round(metrics.recall_score(y_test,y_pred,average=average)*100,3)
		accuracy_list.append(val_acc)
		control+=1
	accuracy_mean=0
	for accuracy in accuracy_list:
			accuracy_mean=float(accuracy_mean)+float(accuracy)
	accuracy_mean=accuracy_mean/len(accuracy_list)
	end=time.time()
	print(
		metrics.classification_report(y_test,y_pred,target_names=target_names),
		"Cross validation score ("+str(k_cross_val)+") : "+str(round(accuracy_mean*100,3)),
		"Accuracy score : "+str(round(metrics.accuracy_score(y_test,y_pred)*100,3)),
		"\tF1_score : "+str(f1_score),
		"\tPrecision : "+str(precision),
		"\tRecall : "+str(recall),
		"\tTime : "+str(round(end-start,3)),
		"\n#######################################################")
	output_file.write(str(f1_score))
	output_file.write("\t")
	output_file.write(str(precision))
	output_file.write("\t")
	output_file.write(str(recall))
	output_file.write("\t")
	output_file.write(str(val_acc*100))
	output_file.write("\t")
	output_file.write(str(round(accuracy_mean*100,3)))
	output_file.write("\t")
	output_file.write(str(val_loss))
	output_file.write("\t")
	output_file.write(str(round(end-start,3)))
	output_file.write("\t")
	output_file.write(str(approach.name))
	output_file.write("\n")
output_file.close()