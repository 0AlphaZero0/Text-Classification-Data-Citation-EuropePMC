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

from keras.utils import normalize

################################################    Variables     #################################################

# Files
dataset_filename="Dataset23.csv"
result_outfile="ResultLogisticR.csv"
# Parameters
k_cross_val=4
average="macro"
gamma="auto"
C=1000
max_iter=1000
tol=1
class_weight={
	0 : 25.,
	1 : 20.,
	2 : 10.,}
skf=StratifiedKFold(
	n_splits=k_cross_val,
	random_state=42)
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
	completeCitation]
target_names=[
	"Background",
	"Creation",
	"Use"]
################################################    Models     #################################################
#
clfLR=LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=max_iter,class_weight=class_weight,tol=tol)
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
data=read_csv(
	filepath_or_buffer=dataset_filename,
	header=0,
	sep=";")
#
data[completeCitation]=data[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]),axis=1)
#
data["Categories_num"]=data.Categories.map({
	"Background":0,
	"Creation":1,
	"Use":2})
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
output_file=codecs.open(result_outfile,'w',encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tf1-scoreCV\tPrecisionCV\tRecallCV\tAccuracyCV\tMethod\tCombination\tToken\tNgram\tLemma\tStem\tTime\n")
for index_vect_list in range(len(vect_list)):
	print(str(index_vect_list+1)+"/"+str(len(vect_list)))
	start=time.time()
	f1_score_list,precision_list,recall_list,accuracy_list=[],[],[],[]
	print(vect_list[index_vect_list][2])

	X_to_train,X_val,y_to_train,y_val=train_test_split(X,y,random_state=42)

	for train_index,test_index in skf.split(X_to_train,y_to_train):
		X_train,X_test=X_to_train.iloc[train_index,],X_to_train.iloc[test_index,]
		y_train,y_test=y_to_train.iloc[train_index],y_to_train.iloc[test_index,]
		vect_X_train,vect_X_test=[],[]
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

		try:
			clfLR.fit(X_train_dtm,y_train)
			y_pred_class=clfLR.predict(X_test_dtm)
		except TypeError:
			clfLR.fit(X_train_dtm.toarray(),y_train)
			y_pred_class=clfLR.predict(X_test_dtm.toarray())

		f1_score=metrics.f1_score(y_test,y_pred_class,average=average)*100
		f1_score_list.append(f1_score)
		precision=metrics.precision_score(y_test,y_pred_class,average=average)*100
		precision_list.append(precision)
		recall=metrics.recall_score(y_test,y_pred_class,average=average)*100
		recall_list.append(recall)
		accuracy=metrics.accuracy_score(y_test,y_pred_class)*100
		accuracy_list.append(accuracy)	
	end=time.time()

	### VALIDATION SET #
	vect_X_val=[]
	vect_X_val.append(vect_list[index_vect_list][0].transform(X_val[[vect_list[index_vect_list][1]]].fillna('').values.reshape(-1)).todense())
	vect_X_val.extend((
		X_val[[Section_num_str]].values,
		X_val[[SubType_num_str]].values,
		X_val[[Figure_num_str]].values))
	X_val_dtm=np.concatenate(vect_X_val,axis=1)
	# y_pred_class_val=clfLR.predict(X_val_dtm)
	result=clfLR.predict_proba(X_val_dtm)
	y_pred_class_val=[]
	for sample in result:
		y_pred_class_val.append(np.argmax(sample))
	f1_score=round(metrics.f1_score(y_val,y_pred_class_val,average=average)*100,3)
	precision=round(metrics.precision_score(y_val,y_pred_class_val,average=average)*100,3)
	recall=round(metrics.recall_score(y_val,y_pred_class_val,average=average)*100,3)
	accuracy=round(metrics.accuracy_score(y_val,y_pred_class_val)*100,3)
	print(
		"\nVALIDATION SET : \n",
		metrics.classification_report(y_val,y_pred_class_val,target_names=target_names),
		"Method : "+str("Logistic Regression"),
		"\nF1_score : "+str(f1_score),
		"\tPrecision : "+str(precision),
		"\tRecall : "+str(recall),
		"\tAccuracy : "+str(accuracy),
		"\tTime : "+str(round(end-start,3))+" sec",
		"\n#######################################################")
	# VALIDATION SET ###

	fold=0
	f1_score_mean,precision_mean,recall_mean,accuracy_mean=0,0,0,0
	while fold < len(f1_score_list):
		f1_score_mean+=f1_score_list[fold]
		precision_mean+=precision_list[fold]
		recall_mean+=recall_list[fold]
		accuracy_mean+=accuracy_list[fold]
		fold+=1
	f1_score_mean=f1_score_mean/len(f1_score_list)
	precision_mean=precision_mean/len(precision_list)
	recall_mean=recall_mean/len(recall_list)
	accuracy_mean=accuracy_mean/len(accuracy_list)
	print(
		metrics.classification_report(y_test,y_pred_class,target_names=target_names),
		"Method : "+str("Logistic Regression"),
		"\nF1_score : " + str(round(f1_score_mean,3)),
		"\tPrecision : " + str(round(precision_mean,3)),
		"\tRecall : " + str(round(recall_mean,3)),
		"\t Accuracy : " + str(round(accuracy_mean,3)),
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
	output_file.write(str(f1_score_mean))
	output_file.write("\t")
	output_file.write(str(precision_mean))
	output_file.write("\t")
	output_file.write(str(recall_mean))
	output_file.write("\t")
	output_file.write(str(accuracy_mean))
	output_file.write("\t")
	output_file.write(str("Logistic Regression"))
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