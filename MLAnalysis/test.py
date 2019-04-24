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
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

################################################    Variables     #################################################
#
filename="Dataset2.csv"
gamma='auto'
C=10
max_iter=10000
class_weight='balanced'
ngram_range=(1,3)
Section_num_str,SubType_num_str,Figure_num_str="Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str="PreCitation","Citation","PostCitation"
featuresList=[Section_num_str,SubType_num_str,Figure_num_str,PreCitation_str,Citation_str,PostCitation_str]
vect_X_train,vect_X_test=[],[]
token='Tokenization'
ngram='N-gram'
lemma='Lemmatization'
stem="Stemming"
extra_features=[token,ngram,lemma,stem]
#
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

##################################################    Class     ###################################################

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl =WordNetLemmatizer()
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
        j=0
        while j<len(src):
            fn(n-1, src[:j], [src[j]] + got, all)
            j=j+1
        return
    all=[]
    i=0
    while i<len(a):
        fn(i,a,[],all)
        i=i+1
    all.append(a)
    return all #a=[1,2,3,4] print(combinations(a))

###################################################    Main     ###################################################
#
combinations_list=combinations(extra_features)
#
vect_list=[
	[TfidfVectorizer(), PreCitation_str, token],
	[TfidfVectorizer(), Citation_str, token],
	[TfidfVectorizer(), PostCitation_str, token],
	[TfidfVectorizer(ngram_range=ngram_range), PreCitation_str, ngram],
	[TfidfVectorizer(ngram_range=ngram_range), Citation_str, ngram],
	[TfidfVectorizer(ngram_range=ngram_range), PostCitation_str, ngram],
	[TfidfVectorizer(tokenizer=LemmaTokenizer()), PreCitation_str, lemma],
	[TfidfVectorizer(tokenizer=LemmaTokenizer()), Citation_str, lemma],
	[TfidfVectorizer(tokenizer=LemmaTokenizer()), PostCitation_str, lemma],
	[TfidfVectorizer(analyzer=stemmed_words), PreCitation_str, stem],
	[TfidfVectorizer(analyzer=stemmed_words), Citation_str, stem],
	[TfidfVectorizer(analyzer=stemmed_words), PostCitation_str, stem]]
#
stemmer = SnowballStemmer('english',ignore_stopwords=True)
analyzer = TfidfVectorizer().build_analyzer()
#
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

for combination in combinations_list:
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

	X_train_dtm = np.concatenate(vect_X_train, axis = 1)

	X_test_dtm = np.concatenate(vect_X_test, axis = 1)

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
			str(round((end-start),3))+" sec",
			"Approaches :"+str(combination))
		print("#######################################################")