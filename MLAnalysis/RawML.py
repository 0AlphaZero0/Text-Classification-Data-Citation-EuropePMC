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
import requests # Allows to make http requests
from sklearn import svm # Allows to use the SVM classification method
from sklearn.feature_extraction.text import CountVectorizer # allow transformation of string in number.
from sklearn import metrics
from sklearn.model_selection import train_test_split 
import sys # Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.

################################################    Variables     #################################################

filename="Dataset1.csv"
gamma='auto'
C=1
clf=svm.SVC(kernel='rbf',gamma=gamma,C=C)
vect=CountVectorizer()

################################################    Functions     #################################################


###################################################    Main     ###################################################

data=pd.read_csv(filename,header=0,sep="\t")
#print(data.shape)
#print(data.head(10))
#print(data.Categories.value_counts())
data["Categories_num"]=data.Categories.map({'Background':0,"ClinicalTrials":1,"Compare":2,"Creation":3,"Unclassifiable":4,"Use":5})
#print(data.head(10))
X=data.Citation
y=data.Categories_num
#print(X.shape)
#print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
X_train_dtm=vect.fit_transform(X_train)
X_test_dtm=vect.transform(X_test)
clf.fit(X_train_dtm,y_train)
y_pred_class=clf.predict(X_test_dtm)
# print(metrics.accuracy_score(y_test,y_pred_class))
# print(metrics.confusion_matrix(y_test,y_pred_class))
# print(X_test[y_pred_class > y_test])# false positive
print(X_test[y_pred_class < y_test])# false negative