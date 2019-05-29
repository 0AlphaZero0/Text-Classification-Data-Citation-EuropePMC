#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 14/05/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import numpy as np # Allows to manipulate the necessary table for sklearn
import time # Allows to measure execution time.

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import read_csv

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
from sklearn.model_selection import StratifiedKFold

################################################    Variables     #################################################

# Files
dataset_filename="Dataset23.csv"
result_outfile="ResultMLparam.csv"
# Parameters
average="macro"
gamma="auto"
C=10
max_iter=10000
class_weight="balanced"
skf=StratifiedKFold(n_splits=4)
ngram_range=(1,3)
# Lemmatizer & Stemmer
lemmatizer=WordNetLemmatizer()
stemmer=SnowballStemmer('english',ignore_stopwords=True)
# Variables & lists of those
token,ngram,lemma,stem="Tokenization","N-gram","Lemmatization","Stemming"
Section_num_str,SubType_num_str,Figure_num_str,NbPaperCitation="Section_num","SubType_num","Figure_num","NbPaperCitation"
PreCitation_str,Citation_str,PostCitation_str,completeCitation="PreCitation","Citation","PostCitation","CompleteCitation"
featuresList=[
	Section_num_str,
	SubType_num_str,
	Figure_num_str,
	# NbPaperCitation,# Add if we work with "number of paper citations" in the data citation sentence
	completeCitation]
target_names=[
	"Background",
	# "Compare",
	"Creation",
	"Use"]
countVectorizerList=[
	"MultinomialNB",
	"ComplementNB",
	"GaussianNB",
	"Random Forest"]
################################################    Models     #################################################
#
clfSVM=svm.LinearSVC(C=C,max_iter=max_iter,class_weight=class_weight)
clfLR=LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=max_iter,class_weight=class_weight)
clfRF=RandomForestClassifier(n_estimators=100,random_state=0) # max_depth=2
clfComplementNB=ComplementNB()
clfGaussianNB=GaussianNB()
clfMultinomialNB=MultinomialNB()
#
clfList=[
	[clfLR,"Logistic-Regression"],
	[clfComplementNB,"ComplementNB"],
	[clfGaussianNB,"GaussianNB"],
	[clfMultinomialNB,"MultinomialNB"],
	[clfRF,"Random-Forest"],
	[clfSVM,"SVM"]]
#
################################################    Functions     #################################################
#
def lemma_word(word):
	"""This function take as args word and return its lemma
	
	Args : 
		- word : (str) a word that could lemmatize by the WordNetLemmatizer from nltk.stem
	
	Return : 
		- word : (str) a lemma of the word gives in args
	"""
	return lemmatizer.lemmatize(word)
#
def lemma_tokenizer(doc):
	""" This function take as args a doc that could be lemmatize.

	Args : 
		- doc : (str) a string that can be tokenize by the word_tokenize of nltk library
	
	Return : 
		- tokens : (list) a list of tokens where each token corresponds to a lemmatized word 
	"""
	tokens = word_tokenize(doc)
	tokens = [lemma_word(t) for t in tokens]
	return tokens
#
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
#
data=read_csv(dataset_filename,header=0,sep=";")
#
data[completeCitation]=data[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]),axis=1)
#
data["Categories_num"]=data.Categories.map({
	"Background":0,
	"Creation":1,
	"Use":2})
	# "Compare":1,
	# "Creation":2,
	# "Use":3})
#
data[Figure_num_str]=data.Figure.map({
	True:0,
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
#
vect_list=[
	[TfidfVectorizer(tokenizer=tokenizer),completeCitation,[token]],
	[TfidfVectorizer(ngram_range=ngram_range),completeCitation,[ngram]],
	[TfidfVectorizer(tokenizer=lemma_tokenizer),completeCitation,[lemma]],
	[TfidfVectorizer(tokenizer=stem_tokenizer),completeCitation,[stem]],
	[TfidfVectorizer(ngram_range=ngram_range,tokenizer=lemma_tokenizer),completeCitation,[ngram,lemma]],
	[TfidfVectorizer(ngram_range=ngram_range,tokenizer=stem_tokenizer),completeCitation,[ngram,stem]]]
#
vect_list_countvect=[
	[CountVectorizer(tokenizer=tokenizer),completeCitation,[token]],
	[CountVectorizer(ngram_range=ngram_range),completeCitation,[ngram]],
	[CountVectorizer(tokenizer=lemma_tokenizer),completeCitation,[lemma]],
	[CountVectorizer(tokenizer=stem_tokenizer),completeCitation,[stem]],
	[CountVectorizer(ngram_range=ngram_range,tokenizer=lemma_tokenizer),completeCitation,[ngram,lemma]],
	[CountVectorizer(ngram_range=ngram_range,tokenizer=stem_tokenizer),completeCitation,[ngram,stem]]]
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
#
output_file=codecs.open(result_outfile,'w',encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tCross-validation-score\tMethod\tCombination\tToken\tNgram\tLemma\tStem\tTime\n")
for index_vect_list in range(len(vect_list)):
	print(str(index_vect_list)+"/"+str(len(vect_list)))
	for clf in clfList:
		start=time.time()
		accuracy_list=[]
		print(vect_list[index_vect_list][2])
		for train_index,test_index in skf.split(X,y):
			X_train,X_test=X.ix[train_index],X.ix[test_index]
			y_train,y_test=y.ix[train_index],y.ix[test_index]
			vect_X_train,vect_X_test=[],[]
			if clf[1] in countVectorizerList:
				vect_tmp=vect_list_countvect[index_vect_list][2]
				vect_X_train.append(vect_list_countvect[index_vect_list][0].fit_transform(X_train[[vect_list_countvect[index_vect_list][1]]].fillna('').values.reshape(-1)).todense())
				vect_X_test.append(vect_list_countvect[index_vect_list][0].transform(X_test[[vect_list_countvect[index_vect_list][1]]].fillna('').values.reshape(-1)).todense())
			else:
				vect_tmp=vect_list[index_vect_list][2]
				vect_X_train.append(vect_list[index_vect_list][0].fit_transform(X_train[[vect_list[index_vect_list][1]]].fillna('').values.reshape(-1)).todense())
				vect_X_test.append(vect_list[index_vect_list][0].transform(X_test[[vect_list[index_vect_list][1]]].fillna('').values.reshape(-1)).todense())

			vect_X_train.extend((
				X_train[[Section_num_str]].values,
				X_train[[SubType_num_str]].values,
				X_train[[Figure_num_str]].values))
			vect_X_test.extend((
				X_test[[Section_num_str]].values,
				X_test[[SubType_num_str]].values,
				X_test[[Figure_num_str]].values))

			X_train_dtm=np.concatenate(vect_X_train,axis=1)
			X_test_dtm=np.concatenate(vect_X_test,axis=1)

			X_train_test=np.concatenate((X_train_dtm,X_test_dtm))
			y_train_test=np.concatenate((y_train,y_test))

			try:
				clf[0].fit(X_train_dtm,y_train)
				y_pred_class=clf[0].predict(X_test_dtm)
			except TypeError:
				clf[0].fit(X_train_dtm.toarray(),y_train)
				y_pred_class=clf[0].predict(X_test_dtm.toarray())

			f1_score=round(metrics.f1_score(y_test,y_pred_class,average=average)*100,3)
			precision=round(metrics.precision_score(y_test,y_pred_class,average=average)*100,3)
			recall=round(metrics.recall_score(y_test,y_pred_class,average=average)*100,3)
			accuracy=round(metrics.accuracy_score(y_test,y_pred_class)*100,3)
			accuracy_list.append(accuracy)
		
		end=time.time()

		accuracy_mean=0
		for accuracy in accuracy_list:
			accuracy_mean=accuracy+accuracy_mean
		accuracy_mean=accuracy_mean/len(accuracy_list)

		print(
			metrics.classification_report(y_test,y_pred_class,target_names=target_names),
			"Method : "+str(clf[1]),
			"\nCross validation score : "+str(round(accuracy_mean,3)),
			"\nAccuracy score : " + str(accuracy),
			"\tF1_score : " + str(f1_score),
			"\tPrecision : " + str(precision),
			"\tRecall : " + str(recall),
			"\tTime : "+str(round(end-start,3))+" sec",
			"\n#######################################################")
		output_file.write(str(f1_score))
		output_file.write("\t")
		output_file.write(str(precision))
		output_file.write("\t")
		output_file.write(str(recall))
		output_file.write("\t")
		output_file.write(str(accuracy))
		output_file.write("\t")
		output_file.write(str(accuracy_mean))
		output_file.write("\t")
		output_file.write(str(clf[1]))
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
		output_file.write("\t")
		output_file.write(str(round(end-start,3)))
		output_file.write("\n")
output_file.close()