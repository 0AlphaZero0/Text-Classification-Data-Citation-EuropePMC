#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
# import codecs # Allows to load a file containing UTF-8 characters
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
# from sklearn. feature_extraction.text import CountVectorizer
# import sys # Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import time # Allows to measure execution time.
# import warnings
from nltk.stem.snowball import SnowballStemmer

################################################    Variables     #################################################

filename="Dataset1.csv"
gamma='auto'
C=10
max_iter=10000
class_weight='balanced'
ngram_range=(1,3)
Section_num_str,SubType_num_str,Figure_num_str="Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str="PreCitation","Citation","PostCitation"
featuresList=[Section_num_str,SubType_num_str,Figure_num_str,PreCitation_str,Citation_str,PostCitation_str]

clfSVM=svm.LinearSVC(C=C,max_iter=max_iter,class_weight=class_weight)
clfLR=LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=max_iter,class_weight=class_weight)
clfRF = RandomForestClassifier(n_estimators=100,random_state=0) # max_depth=2
clfBernoulliNB=BernoulliNB()
clfComplementNB=ComplementNB()
clfGaussianNB=GaussianNB()
clfMultinomialNB= MultinomialNB()
clfList=[[clfLR,"Logistic Regression"],
	[clfBernoulliNB,"BernoulliNB"],
	[clfComplementNB,"ComplementNB"],
	[clfGaussianNB,"GaussianNB"],
	[clfMultinomialNB,"MultinomialNB"],
	[clfRF,"Random Forest"],
	[clfSVM,"SVM"]]

pre_vect=TfidfVectorizer()
ngram_pre_vect=TfidfVectorizer(ngram_range=ngram_range)
citation_vect=TfidfVectorizer()
ngram_citation_vect=TfidfVectorizer(ngram_range=ngram_range)
post_vect=TfidfVectorizer()
ngram_post_vect=TfidfVectorizer(ngram_range=ngram_range)
stemmer = SnowballStemmer('english',ignore_stopwords=True)
analyzer = TfidfVectorizer().build_analyzer()
################################################    Functions     #################################################

def stemmed_words(doc):
	return (stemmer.stem(w) for w in analyzer(doc))

stem_citation_vect=TfidfVectorizer(analyzer=stemmed_words)
stem_precitation_vect=TfidfVectorizer(analyzer=stemmed_words)
stem_postcitation_vect=TfidfVectorizer(analyzer=stemmed_words)

###################################################    Main     ###################################################

data=pd.read_csv(filename,header=0,sep="\t")
#

data["Categories_num"]=data.Categories.map(
	{"Background":0,
	"ClinicalTrials":1,
	"Compare":2,
	"Creation":3,
	"Unclassifiable":4,
	"Use":5})
#
data[Figure_num_str]=data.Figure.map(
	{True:0,
	False:1})
#
sectionDict={}
index=1
for section in data.Section:
	if section not in sectionDict:
		sectionDict[section]=index
		index+=1
data[Section_num_str]=data.Section.map(sectionDict)
#
subTypeDict={}
index=1
for subType in data.SubType:
	if subType not in subTypeDict:
		subTypeDict[subType]=index
		index+=1
data[SubType_num_str]=data.SubType.map(subTypeDict)
#
##################################################################
#
X=data[featuresList]
y=data.Categories_num

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

# Citations only #
# X_train_dtm = citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense()
# X_test_dtm = citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense()

# Stemming Citation only #
# X_train_dtm = stem_citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense()
# X_test_dtm = stem_citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense()

# Ngram & Token & Stemming with Citation #
# X_train_dtm= np.concatenate(
#     (X_train[[Section_num_str]].values,
#     X_train[[SubType_num_str]].values,
#     X_train[[Figure_num_str]].values,citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense(),
# 	stem_citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense(),
# 	ngram_citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense()),
#     axis=1
# 	)

# X_test_dtm= np.concatenate(
#     (X_test[[Section_num_str]].values,
#     X_test[[SubType_num_str]].values,
#     X_test[[Figure_num_str]].values,citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense(),
# 	stem_citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense(),
# 	ngram_citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense()),
#     axis=1
# )

X_train_dtm= np.concatenate(
    (X_train[[Section_num_str]].values,
    X_train[[SubType_num_str]].values,
    X_train[[Figure_num_str]].values,
    pre_vect.fit_transform(X_train[[PreCitation_str]].fillna('').values.reshape(-1)).todense(),
	ngram_pre_vect.fit_transform(X_train[[PreCitation_str]].fillna('').values.reshape(-1)).todense(),
	stem_precitation_vect.fit_transform(X_train[[PreCitation_str]].fillna('').values.reshape(-1)).todense(),
    citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense(),
	ngram_citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense(),
	stem_citation_vect.fit_transform(X_train[[Citation_str]].fillna('').values.reshape(-1)).todense(),
    post_vect.fit_transform(X_train[[PostCitation_str]].fillna('').values.reshape(-1)).todense(),
	ngram_post_vect.fit_transform(X_train[[PostCitation_str]].fillna('').values.reshape(-1)).todense(),
	stem_postcitation_vect.fit_transform(X_train[[PostCitation_str]].fillna('').values.reshape(-1)).todense()),
    axis=1
	)
X_test_dtm= np.concatenate(
    (X_test[[Section_num_str]].values,
    X_test[[SubType_num_str]].values,
    X_test[[Figure_num_str]].values,
    pre_vect.transform(X_test[[PreCitation_str]].fillna('').values.reshape(-1)).todense(),
	ngram_pre_vect.transform(X_test[[PreCitation_str]].fillna('').values.reshape(-1)).todense(),
	stem_precitation_vect.transform(X_test[[PreCitation_str]].fillna('').values.reshape(-1)).todense(),
    citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense(),
	ngram_citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense(),
	stem_citation_vect.transform(X_test[[Citation_str]].fillna('').values.reshape(-1)).todense(),
    post_vect.transform(X_test[[PostCitation_str]].fillna('').values.reshape(-1)).todense(),
	ngram_post_vect.transform(X_test[[PostCitation_str]].fillna('').values.reshape(-1)).todense(),
	stem_postcitation_vect.transform(X_test[[PostCitation_str]].fillna('').values.reshape(-1)).todense()),
    axis=1
)
##################################################################
for clf in clfList:
	start=time.time()
	try:
		print (X_train_dtm.shape,"X_train shape")
		print (y_train.shape,"y_train shape")
		clf[0].fit(X_train_dtm,y_train)
		y_pred_class=clf[0].predict(X_test_dtm)
	except TypeError:
		clf[0].fit(X_train_dtm.toarray(),y_train)
		y_pred_class=clf[0].predict(X_test_dtm.toarray())
	end=time.time()
	target_names=["Background","ClinicalTrials","Compare","Creation","Unclassifiable","Use"]
	print(metrics.classification_report(y_test,y_pred_class,target_names=target_names),
		metrics.accuracy_score(y_test,y_pred_class),
		"\t",
		clf[1],
		"\t",
		str(round((end-start),3))+" sec")
	print("#######################################################")