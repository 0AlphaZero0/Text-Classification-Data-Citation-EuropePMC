#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################

import tensorflow as tf
import numpy as np
import pandas as pd
import time

from sklearn import svm # Allows to use the SVM classification method
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

completeCitation = "CompleteCitation"
token = 'Tokenization'
ngram = 'N-gram'
lemma = 'Lemmatization'
stem = "Stemming"
filename = "Dataset2.csv"
gamma = 'auto'
Section_num_str,SubType_num_str,Figure_num_str = "Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str = "PreCitation","Citation","PostCitation"
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

for combination in combinations_list:
	for clf in clfList:
		vect_X_train,vect_X_test = [],[]
		if clf[1] in countVectorizerList:
			for vect in vect_list_countvectorizer:
				if vect[2] in combination:
					vect_X_train.append(vect[0].fit_transform(X_train[[vect[1]]].fillna('').values.reshape(-1)).todense())
					vect_X_test.append(vect[0].transform(X_test[[vect[1]]].fillna('').values.reshape(-1)).todense())
		else:
			for vect in vect_list:
				if vect[2] in combination:
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
		X_train_dtm = tf.keras.utils.normalize(np.concatenate(vect_X_train, axis = 1), axis = 1)
		X_test_dtm = tf.keras.utils.normalize(np.concatenate(vect_X_test, axis = 1), axis = 1)
		X_train_test = np.concatenate((X_train_dtm,X_test_dtm))
		y_train_test = np.concatenate((y_train,y_test))
		
		model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=X_train_dtm.shape),
		tf.keras.layers.Dense(128, activation = tf.nn.relu),
		tf.keras.layers.Dense(128, activation = tf.nn.relu),
		tf.keras.layers.Dense(128, activation = tf.nn.relu),
		tf.keras.layers.Dense(128, activation = tf.nn.relu),
		tf.keras.layers.Dense(10, activation = tf.nn.softmax),
		])
		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=['accuracy'])
		model.fit(X_train_dtm, y_train, epochs = 3)