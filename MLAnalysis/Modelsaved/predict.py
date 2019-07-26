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
dataset_to_predict="Result.csv"
result_outfile="Predictions.csv"

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
clfLR=joblib.load("LRDCS.joblib")
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
# Loading data to predict
data_to_predict=read_csv(
	filepath_or_buffer=dataset_to_predict,
	header=0,
	sep="\t")
# Preprocess
data_to_predict=data_to_predict.fillna('')
data_to_predict[completeCitation]=data_to_predict[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]),axis=1)
data_to_predict[Figure_num_str]=data_to_predict.Figure.map({
	True:0,
	False:1})
sectionDict=joblib.load("sectionDict.joblib")
data_to_predict[Section_num_str]=data_to_predict.Section.map(sectionDict)
subTypeDict=joblib.load("subTypeDict.joblib")
data_to_predict[SubType_num_str]=data_to_predict.SubType.map(subTypeDict)
#
##################################################################
###   TO PREDICT   ###
X_test=data_to_predict[featuresList]
#
vect_X_test=[]
vectorizer=joblib.load("Vectorizer.joblib")
vect_X_test.append(vectorizer.transform(X_test[[completeCitation]].fillna('').values.reshape(-1)).todense())
vect_X_test.extend((
	X_test[[Section_num_str]].values,
	X_test[[SubType_num_str]].values,
	X_test[[Figure_num_str]].values))
X_test_dtm=np.concatenate(vect_X_test,axis=1)
X_test_dtm=np.nan_to_num(X_test_dtm)
####    ####
clfLR=joblib.load("LRDCS.joblib")
result=clfLR.predict_proba(X_test_dtm)
y_pred_class=[]
for sample in result:
	y_pred_class.append(np.argmax(sample))
####    ####
dict_cat_name={
	0:"Background",
	1:"Creation",
	2:"Use"}
output_file=codecs.open(result_outfile,'w',encoding='utf8')
output_file.write("PMCID\tAccessionNb\tSection\tSubType\tFigure\tCategories\tPreCitation\tCitation\tPostCitation\tBackground\tCreation\tUse\n")
for index,row in data_to_predict.iterrows():
	output_file.write(str(row["PMCID"]))
	output_file.write("\t")
	output_file.write(str(row["AccessionNb"]))
	output_file.write("\t")
	output_file.write(str(row["Section"]))
	output_file.write("\t")
	output_file.write(str(row["SubType"]))
	output_file.write("\t")
	output_file.write(str(row["Figure"]))
	output_file.write("\t")
	output_file.write(str(dict_cat_name[y_pred_class[index]]))
	output_file.write("\t")
	output_file.write(str(row["PreCitation"]))
	output_file.write("\t")
	output_file.write(str(row["Citation"]))
	output_file.write("\t")
	output_file.write(str(row["PostCitation"]))
	output_file.write("\t")
	output_file.write(str(result[index][0]))
	output_file.write("\t")
	output_file.write(str(result[index][1]))
	output_file.write("\t")
	output_file.write(str(result[index][2]))
	output_file.write("\n")
output_file.close()