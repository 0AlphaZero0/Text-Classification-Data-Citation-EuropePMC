#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################

import codecs
import numpy as np
from numpy import concatenate
from numpy import argmax
from pandas import read_csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import normalize
from keras import backend
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

##################################################    Variables     ###################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


filename = "Dataset2.csv"
result_output="ResultDLparam.csv"
average="macro" # binary | micro | macro | weighted | samples
class_weight = {
	0 : 15.,
	1 : 50.,
	2 : 15.,
	3 : 10.}
skf = StratifiedKFold(n_splits=4)
epochs = 5
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english',ignore_stopwords = True)
activation_input_node = 'relu'
node1 = 128
activation_node1 = 'relu'
node2 = 128
activation_node2 = 'relu'
output_node = 4
activation_output_node='softmax'
ngram_range = (1,3)
token,ngram,lemma,stem = "Tokenization","N-gram","Lemmatization","Stemming"
Section_num_str,SubType_num_str,Figure_num_str = "Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str,completeCitation = "PreCitation","Citation","PostCitation","CompleteCitation"
featuresList = [
	Section_num_str,
	SubType_num_str,
	Figure_num_str,
	completeCitation]
target_names = [
	"Background",
	"Compare",
	"Creation",
	"Use"]

################################################    Functions     #################################################

def lemma_word(word):
	return lemmatizer.lemmatize(word)

def lemma_tokenizer(doc):
	tokens = word_tokenize(doc)
	tokens = [lemma_word(t) for t in tokens]
	return tokens

def stem_word(word):
	return stemmer.stem(word)

def stem_tokenizer(doc):
	tokens = word_tokenize(doc)
	tokens = [stem_word(t) for t in tokens]
	return tokens

def tokenizer(doc):
	tokens = word_tokenize(doc)
	print (tokens)
	return tokens

###################################################    Main     ###################################################
#
data = read_csv(filename,header = 0,sep = "\t")
#
data[completeCitation] = data[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]), axis = 1)
#
data["Categories_num"] = data.Categories.map(
	{"Background":0,
	"Compare":1,
	"Creation":2,
	"Use":3})
#
data[Figure_num_str] = data.Figure.map(
	{True:0,
	False:1})
#
sectionDict = {}
index_section = 1
for section in data.Section:
	if section not in sectionDict:
		sectionDict[section] = index_section
		index_section+=1
data[Section_num_str] = data.Section.map(sectionDict)
#
subTypeDict = {}
index_subtype = 1
for subType in data.SubType:
	if subType not in subTypeDict:
		subTypeDict[subType] = index_subtype
		index_subtype+=1
data[SubType_num_str] = data.SubType.map(subTypeDict)
#
##################################################################
#
X = data[featuresList]
y = data.Categories_num

##################################"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
import sys
def sizeof_fmt(num, suffix='B'):
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			return "%3.1f%s%s" % (num, unit, suffix)
		num /= 1024.0
	return "%.1f%s%s" % (num, 'Yi', suffix)
##################################"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

vect_list = [
	[TfidfVectorizer(tokenizer = tokenizer), completeCitation, [token]],
	[TfidfVectorizer(ngram_range = ngram_range), completeCitation, [ngram]],
	[TfidfVectorizer(tokenizer = lemma_tokenizer), completeCitation, [lemma]],
	[TfidfVectorizer(tokenizer = stem_tokenizer), completeCitation, [stem]],
	[TfidfVectorizer(ngram_range = ngram_range, tokenizer = lemma_tokenizer),completeCitation,[ngram,lemma]],
	[TfidfVectorizer(ngram_range = ngram_range, tokenizer = stem_tokenizer),completeCitation,[ngram,stem]]]

output_file=codecs.open(result_output,'w',encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tCross-score\tLoss\tCombination\tToken\tNgram\tLemma\tStem\n")
for vect in vect_list:
	accuracy_list=[]
	for train_index, test_index in skf.split(X,y):
		print (vect[2])
		X_train, X_test = X.ix[train_index], X.ix[test_index]
		y_train, y_test = y.ix[train_index], y.ix[test_index]
		# print(str(list(set(y_train)))+" TRAIN\n"+str(list(set(y_test)))+" TEST\n")
		
		vect_X_train, vect_X_test = [], []
		vect_X_train.append(vect[0].fit_transform(X_train[[vect[1]]].fillna('').values.reshape(-1)).todense())
		vect_X_test.append(vect[0].transform(X_test[[vect[1]]].fillna('').values.reshape(-1)).todense())

		vect_X_train.extend((
			X_train[[Section_num_str]].values,
			X_train[[SubType_num_str]].values,
			X_train[[Figure_num_str]].values))
		vect_X_test.extend((
			X_test[[Section_num_str]].values,
			X_test[[SubType_num_str]].values,
			X_test[[Figure_num_str]].values))

		# print ("vect_X_train",len(vect_X_train))
		# print ("vect_X_test",len(vect_X_test))

		X_train_dtm = concatenate(vect_X_train, axis = 1)
		X_train_dtm = normalize(X_train_dtm, axis = 1)
		# print (X_train_dtm)
		# print(np.sum(X_train_dtm[0]))

		X_test_dtm = concatenate(vect_X_test, axis = 1)
		X_test_dtm = normalize(X_test_dtm, axis = 1)

		# print ("X_train_dtm shape",X_train_dtm.shape)
		# print ("X_test_dtm shape",X_test_dtm.shape)
		model = Sequential([
			Dense(1280, activation = activation_input_node, input_dim=X_train_dtm[0].shape[1]),
			Dense(node1, activation = activation_node1),
			Dense(node2, activation = activation_node2),
			Dense(output_node, activation = activation_output_node),
			])

		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=['accuracy'])

		model.fit(
			X_train_dtm,
			y_train,
			epochs = epochs,
			batch_size = 20,
			class_weight = class_weight)

		val_loss, val_acc = model.evaluate(X_test_dtm, y_test)

		accuracy_list.append(val_acc)

	##################################"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),key= lambda x: -x[1])[:10]:
			print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))
	for name, size in sorted(((name, sys.getsizeof(value)) for name,value in globals().items()),key= lambda x: -x[1])[:10]:
			print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))
	##################################"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	
	result = model.predict(X_test_dtm)
	
	y_pred=[]
	for sample in result:
		y_pred.append(argmax(sample))

	f1_score = round(metrics.f1_score(y_test, y_pred, average = average)*100,3)
	precision = round(metrics.precision_score(y_test, y_pred, average = average)*100,3)
	recall = round(metrics.recall_score(y_test, y_pred, average = average)*100,3)
		

	accuracy_mean = 0
	for accuracy in accuracy_list:
		accuracy_mean += accuracy
	accuracy_mean = accuracy_mean/len(accuracy_list)
	print(
		metrics.classification_report(y_test,y_pred,target_names = target_names),
		"Cross score : "+str(round(accuracy_mean*100,3)),
		"Accuracy score : "+str(round(metrics.accuracy_score(y_test,y_pred)*100,3)),
		"\tF1_score : "+str(f1_score),
		"\tPrecision : "+str(precision),
		"\tRecall : "+str(recall),
		"\n#######################################################")

	output_file.write(str(f1_score))
	output_file.write("\t")
	output_file.write(str(precision))
	output_file.write("\t")
	output_file.write(str(recall))
	output_file.write("\t")
	output_file.write(str(round(val_acc*100,3)))
	output_file.write("\t")
	output_file.write(str(round(accuracy_mean*100,3)))
	output_file.write("\t")
	output_file.write(str(round(val_loss,3)))
	output_file.write("\t")
	output_file.write(str(vect[2]))
	output_file.write("\t")
	if token in vect[2]:
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\t")
	if ngram in vect[2]:
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\t")
	if lemma in vect[2]:
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\t")
	if stem in vect[2]:
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\n")
	
	f1_score = None
	precision = None
	recall = None
	X_train_dtm = None
	X_test_dtm = None
	X_train = None
	X_test = None
	y_train=None
	y_test=None
	model = None
	stemmer = None
	vect_list = None
	vect_X_train,vect_X_test = None,None
	vect_tmp = None
	val_loss,val_acc=None,None
	backend.clear_session()
		