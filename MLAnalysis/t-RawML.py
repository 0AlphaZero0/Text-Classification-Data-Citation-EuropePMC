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
import sys # Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.

"""
"""

################################################    Variables     #################################################

filename="Dataset1.csv"
gamma='auto'
C=1
clf=svm.SVC(kernel='rbf',gamma=gamma,C=C)
vect=CountVectorizer()

################################################    Functions     #################################################

def loadDataset(filename):
    dataset=[]
    file=codecs.open(filename,"r",encoding="utf-8")
    for line in file.readlines()[1:]:
        dataset.append(line.split("\t"))
    file.close()
    return dataset

def split(dataset):
    tmp=copy.deepcopy(dataset)
    testSet=[]
    while len(tmp)>nbOfCitations-(nbOfCitations/4):
        randomIndex=random.randint(0,len(tmp)-1)
        testSet.append(tmp.pop(randomIndex))
    trainingSet=tmp
    return trainingSet,testSet

def splitAnnotationFromData(data):
    dataAnnotation=[]
    for citation in data:
        dataAnnotation.append(citation.pop(5))
    return dataAnnotation,data


###################################################    Main     ###################################################

dataset=loadDataset(filename)
nbOfCitations=len(dataset)
#
tmpSplit=split(dataset)
trainingSet=tmpSplit[0]
testSet=tmpSplit[1]
nbOfCitationsTrainingSet=len(trainingSet)
nbOfCitationsTestSet=len(testSet)
#
tmpSplit=splitAnnotationFromData(trainingSet)
trainingSet=tmpSplit[1]
annotationTrainingSet=tmpSplit[0]
#
tmpSplit=splitAnnotationFromData(testSet)
testSet=tmpSplit[1]
annotationTestSet=tmpSplit[0]
# Raw dataset and only citation /pre-citation+post-citation
datasetOnlyCitation=[]
for citation in dataset:
    pass
vect.fit()








# clf.fit(x_train,y_train)
# result=clf.predict(y_test)
# index=0
# success=0
# while index<len(y_test):
#     if result[index]==y_test[index]:
#         success+=1
#     index+=1
# print(float(success/len(y_test)*100))