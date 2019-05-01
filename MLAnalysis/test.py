#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
# import copy # Allows to duplicate objects
import numpy as np # Allows to manipulate the necessary table for sklearn
# import os # Allows to modify some things on the os
import pandas as pd # Allows some data manipulations
# import random # Allows to use random variables
# import requests # Allows to make http requests
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import metrics
from sklearn import svm # Allows to use the SVM classification method
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
# import sys # Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import time # Allows to measure execution time.
# import warnings
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

################################################    Variables     #################################################
#
result_output="ResultML.csv"
filename = "Dataset2.csv"
token,ngram,lemma,stem = "Tokenization","N-gram","Lemmatization","Stemming"
Section_num_str,SubType_num_str,Figure_num_str = "Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str,completeCitation = "PreCitation","Citation","PostCitation","CompleteCitation"
average = "macro"
gamma = 'auto'
C = 10
max_iter = 10000
class_weight = 'balanced'
ngram_range = (1,3)
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
extra_features = [
	token,
	ngram,
	lemma,
	stem]
countVectorizerList = [
	"MultinomialNB",
	"ComplementNB",
	"GaussianNB",
	"Random Forest"]
#
clfSVM = svm.LinearSVC(C = C,max_iter = max_iter,class_weight = class_weight)
clfLR = LogisticRegression(C = C,solver = 'lbfgs',multi_class = 'multinomial',max_iter = max_iter,class_weight = class_weight)
clfRF  =  RandomForestClassifier(n_estimators = 100,random_state = 0) # max_depth = 2
clfComplementNB = ComplementNB()
clfGaussianNB = GaussianNB()
clfMultinomialNB =  MultinomialNB()
#
clfList = [[clfLR,"Logistic Regression"],
	[clfComplementNB,"ComplementNB"],
	[clfGaussianNB,"GaussianNB"],
	[clfMultinomialNB,"MultinomialNB"],
	[clfRF,"Random Forest"],
	[clfSVM,"SVM"]]

##################################################    Class     ###################################################

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

################################################    Functions     #################################################

def stemmed_words(doc):
	return (stemmer.stem(w) for w in analyzer(doc))

def combinations(a):
	def fn(n,src,got,all):
		if n==0:
			if len(got)>0:
				all.append(got)
			return
		j = 0
		while j<len(src):
			fn(n-1, src[:j], [src[j]] + got, all)
			j = j+1
		return
	all = []
	i = 0
	while i<len(a):
		fn(i,a,[],all)
		i = i+1
	all.append(a)
	return all #a = [1,2,3,4] print(combinations(a))

###################################################    Main     ###################################################
#
vect_list = [
	[TfidfVectorizer(), completeCitation, token],
	[TfidfVectorizer(ngram_range = ngram_range), completeCitation, ngram],
	[TfidfVectorizer(tokenizer = LemmaTokenizer()), completeCitation, lemma],
	[TfidfVectorizer(analyzer = stemmed_words), completeCitation, stem]]
vect_list_countvectorizer = [
	[CountVectorizer(), completeCitation, token],
	[CountVectorizer(ngram_range = ngram_range), completeCitation, ngram],
	[CountVectorizer(tokenizer = LemmaTokenizer()), completeCitation, lemma],
	[CountVectorizer(analyzer = stemmed_words), completeCitation, stem]]
#
stemmer = SnowballStemmer('english',ignore_stopwords = True)
analyzer = TfidfVectorizer().build_analyzer()
#
data = pd.read_csv(filename,header = 0,sep = "\t")
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
index = 1
for section in data.Section:
	if section not in sectionDict:
		sectionDict[section] = index
		index+=1
data[Section_num_str] = data.Section.map(sectionDict)
#
subTypeDict = {}
index = 1
for subType in data.SubType:
	if subType not in subTypeDict:
		subTypeDict[subType] = index
		index+=1
data[SubType_num_str] = data.SubType.map(subTypeDict)
#
##################################################################
#
X = data[featuresList]
y = data.Categories_num

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)

combinations_list = combinations(extra_features)

output_file=codecs.open(result_output,'w',encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tCombination\tToken\tNgram\tLemma\tStem\n")
for combination in combinations_list:
	for clf in clfList:
		vect_X_train,vect_X_test = [],[]
		vect_tmp=[]
		if clf[1] in countVectorizerList:
			for vect in vect_list_countvectorizer:
				if vect[2] in combination:
					vect_tmp.append(vect[2])
					vect_X_train.append(vect[0].fit_transform(X_train[[vect[1]]].fillna('').values.reshape(-1)).todense())
					vect_X_test.append(vect[0].transform(X_test[[vect[1]]].fillna('').values.reshape(-1)).todense())
		else:
			for vect in vect_list:
				if vect[2] in combination:
					vect_tmp.append(vect[2])
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

		X_train_dtm = np.concatenate(vect_X_train, axis = 1)

		X_test_dtm = np.concatenate(vect_X_test, axis = 1)

		X_train_test = np.concatenate((X_train_dtm,X_test_dtm))

		y_train_test = np.concatenate((y_train,y_test))
		##################################################################
		start = time.time()
		try:
			print ("#######################################################")
			print (X_train_dtm.shape,"X_train shape")
			print (y_train.shape,"y_train shape")
			clf[0].fit(X_train_dtm,y_train)
			y_pred_class = clf[0].predict(X_test_dtm)
			scores=cross_val_score(clf[0], X_train_test, y_train_test, cv = 5)
		except TypeError:
			clf[0].fit(X_train_dtm.toarray(),y_train)
			y_pred_class = clf[0].predict(X_test_dtm.toarray())
			scores=cross_val_score(clf[0], X_train_test, y_train_test, cv = 5)
		end = time.time()

		f1_score = round(metrics.f1_score(y_test, y_pred_class, average = average)*100,3)
		precision = round(metrics.precision_score(y_test, y_pred_class, average = average)*100,3)
		recall = round(metrics.recall_score(y_test, y_pred_class, average = average)*100,3)
		accuracy = round(metrics.accuracy_score(y_test,y_pred_class)*100,3)

		print(
			metrics.classification_report(y_test,y_pred_class,target_names = target_names),
			"Accuracy score : " + str(accuracy),
			"\tF1_score : " + str(f1_score),
			"\tPrecision : " + str(precision),
			"\tRecall : " + str(recall),
			"\n#######################################################")
		output_file.write(str(f1_score))
		output_file.write("\t")
		output_file.write(str(precision))
		output_file.write("\t")
		output_file.write(str(recall))
		output_file.write("\t")
		output_file.write(str(accuracy))
		output_file.write("\t")
		output_file.write(str(vect_tmp))
		output_file.write("\t")
		if token in vect_tmp:
			output_file.write("True")
		else:
			output_file.write("False")
		output_file.write("\t")
		if ngram in vect_tmp:
			output_file.write("True")
		else:
			output_file.write("False")
		output_file.write("\t")
		if lemma in vect_tmp:
			output_file.write("True")
		else:
			output_file.write("False")
		output_file.write("\t")
		if stem in vect_tmp:
			output_file.write("True")
		else:
			output_file.write("False")
		output_file.write("\n")