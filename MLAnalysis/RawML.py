#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import copy # Allows to duplicate objects
import numpy as np # Allows to manipulate the necessary table for sklearn
import os # Allows to modify some things on the os
import pandas as pd # Allows some data manipulations
import random # Allows to use random variables
# import requests # Allows to make http requests
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
import sys # Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import time # Allows to measure execution time.
import warnings

################################################    Variables     #################################################

filename="Dataset1.csv"
gamma='auto'
C=10
max_iter=10000
class_weight='balanced'
clfSVM=svm.LinearSVC(C=C,max_iter=max_iter,class_weight=class_weight)
clfLR=LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=max_iter,class_weight=class_weight)
clfRF = RandomForestClassifier(n_estimators=100,random_state=0) # max_depth=2
clfBernoulliNB=BernoulliNB()
clfComplementNB=ComplementNB()
clfGaussianNB=GaussianNB()
clfMultinomialNB= MultinomialNB()
pre_vect=TfidfVectorizer()
citation_vect=TfidfVectorizer()
post_vect=TfidfVectorizer()
featuresList=['Section_num','SubType_num','Figure_num','PreCitation','Citation','PostCitation']
################################################    Functions     #################################################

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
    return all

###################################################    Main     ###################################################

clfList=[[clfLR,"Logistic Regression"],
        [clfBernoulliNB,"BernoulliNB"],
        [clfComplementNB,"ComplementNB"],
        [clfGaussianNB,"GaussianNB"],
        [clfMultinomialNB,"MultinomialNB"],
        [clfRF,"Random Forest"],
        [clfSVM,"SVM"]]

data=pd.read_csv(filename,header=0,sep="\t")

data["Categories_num"]=data.Categories.map({"Background":0,
                                            "ClinicalTrials":1,
                                            "Compare":2,
                                            "Creation":3,
                                            "Unclassifiable":4,
                                            "Use":5})
data["Figure_num"]=data.Figure.map({'True':1,
                                    'False':0})
sectionDict={}
index=0
for section in data.Section:
    if section not in sectionDict:
        sectionDict[section]=index
        index+=1
data["Section_num"]=data.Section.map(sectionDict)
subTypeDict={}
index=0
for subType in data.SubType:
    if subType not in subTypeDict:
        subTypeDict[subType]=index
        index+=1
data["SubType_num"]=data.SubType.map(subTypeDict)

##################################################################
"""
X=data.Citation
y=data.Categories_num

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

X_train_dtm=vect.fit_transform(X_train)
X_test_dtm=vect.transform(X_test)
"""
X=data[featuresList]
y=data.Categories_num

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
print(X_train[['Section_num']].shape)
print(X_train[['SubType_num']].shape)
print(X_train[['Figure_num']].shape)
print(X_train[['PreCitation']].shape)
print(X_train[['PreCitation']].values.reshape(-1))
print(pre_vect.fit_transform(X_train[['PreCitation']].fillna('').values.reshape(-1)).todense().shape)
print(citation_vect.fit_transform(X_train[['Citation']].fillna('').values.reshape(-1)).todense().shape)
print(post_vect.fit_transform(X_train[['PostCitation']].fillna('').values.reshape(-1)).todense().shape)

all = np.concatenate(
    (X_train[['Section_num']].values,
    X_train[['SubType_num']].values,
    X_train[['Figure_num']].values,
    pre_vect.fit_transform(X_train[['PreCitation']].fillna('').values.reshape(-1)).todense(),
    citation_vect.fit_transform(X_train[['Citation']].fillna('').values.reshape(-1)).todense(),
    post_vect.fit_transform(X_train[['PostCitation']].fillna('').values.reshape(-1)).todense()),
    axis=1
)

print(all.shape)
raise
X_train_dtm=np.concatenate((X_train[['Section_num']],X_train[['SubType_num']],X_train[['Figure_num']],vect.fit_transform(X_train[['PreCitation']]),vect.fit_transform(X_train[['Citation']]),vect.fit_transform(X_train[['PostCitation']])))
X_test_dtm=np.concatenate((X_test[['Section_num']],X_test[['SubType_num']],X_test[['Figure_num']],vect.fit_transform(X_test[['PreCitation']]),vect.fit_transform(X_test[['Citation']]),vect.fit_transform(X_test[['PostCitation']])))
##################################################################

for clf in clfList:
    start=time.time()
    try:
        print (X_train_dtm.shape)
        print (y_train.shape)
        clf[0].fit(X_train_dtm,y_train)
        # print (clf.feature_importance)
        y_pred_class=clf[0].predict(X_test_dtm)
    except TypeError:
        clf[0].fit(X_train_dtm.toarray(),y_train)
        y_pred_class=clf[0].predict(X_test_dtm.toarray())
        # print (clf.feature_importance)
    end=time.time()
    target_names=["Background","ClinicalTrials","Compare","Creation","Unclassifiable","Use"]
    print(metrics.classification_report(y_test,y_pred_class,target_names=target_names),metrics.accuracy_score(y_test,y_pred_class),"\t", clf[1],"\t",str(round((end-start),3))+" sec")
    print("#######################################################")