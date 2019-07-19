#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 19/07/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import joblib
import numpy as np # Allows to manipulate the necessary table for sklearn

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

from pandas import read_csv

from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.linear_model import LogisticRegression

################################################    Variables     #################################################

# Files
dataset_filename="Dataset23.csv"
vecto_outfile="Vectorizer.joblib"
model_outfile="LRDCS.joblib" # Logistic Regression Data Citation Stemming

# Stemmer
stemmer=SnowballStemmer('english',ignore_stopwords=True)
Section_num_str,SubType_num_str,Figure_num_str,NbPaperCitation="Section_num","SubType_num","Figure_num","NbPaperCitation"
PreCitation_str,Citation_str,PostCitation_str,completeCitation="PreCitation","Citation","PostCitation","CompleteCitation"
featuresList=[
	Section_num_str,
	SubType_num_str,
	Figure_num_str,
	completeCitation]
target_names=[
	"Background",
	"Creation",
	"Use"]

################################################    Models     #################################################
#
clfLR=LogisticRegression(
    C=1000,
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=1000,
    class_weight={
	    0 : 25.,
	    1 : 20.,
	    2 : 10.,},
        tol=1)
#
################################################    Functions     #################################################
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
	tokens = word_tokenize(doc)
	tokens = [stem_word(t) for t in tokens]
	return tokens
#
def tokenizer(doc):
	""" This function take as args a doc that could be tokenize.

	Args :
		- doc : (str) a string that can be tokenize by the word_tokenize of nltk library
	
	Return : 
		- tokens : (list) a list of tokens where each token corresponds to a word
	"""
	tokens = word_tokenize(doc)
	return tokens
#
###################################################    Main     ###################################################
# Loading dataset
data=read_csv(
	filepath_or_buffer=dataset_filename,
	header=0,
	sep="\t")
# Preprocess
data[completeCitation]=data[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]),axis=1)
data["Categories_num"]=data.Categories.map({
	"Background":0,
	"Creation":1,
	"Use":2})
data[Figure_num_str]=data.Figure.map({
	True:0,
	False:1})
sectionDict,index={},1
for section in data.Section:
	if section not in sectionDict:
		sectionDict[section]=index
		index+=1
data[Section_num_str]=data.Section.map(sectionDict)
subTypeDict,index={},1
for subType in data.SubType:
	if subType not in subTypeDict:
		subTypeDict[subType]=index
		index+=1
data[SubType_num_str]=data.SubType.map(subTypeDict)
###   TRAIN SET   ###
X_train=data[featuresList]
y_train=data.Categories_num
vectorizer=TfidfVectorizer(tokenizer=stem_tokenizer)
#
vect_X_train=[]
vect_X_train.append(vectorizer.fit_transform(X_train[[completeCitation]].fillna('').values.reshape(-1)).todense())
vect_X_train.extend((
	X_train[[Section_num_str]].values,
	X_train[[SubType_num_str]].values,
	X_train[[Figure_num_str]].values))
X_train_dtm=np.concatenate(vect_X_train,axis=1)
####    ####    ####    ####    ####    ####    ####    ####    ####    ####
clfLR.fit(X_train_dtm,y_train)
########    SAVE    ########
joblib.dump(
    sectionDict,
    "sectionDict.joblib")
joblib.dump(
    subTypeDict,
    "subTypeDict.joblib")
joblib.dump(
    vectorizer,
    vecto_outfile)
joblib.dump(
    clfLR,
    model_outfile)