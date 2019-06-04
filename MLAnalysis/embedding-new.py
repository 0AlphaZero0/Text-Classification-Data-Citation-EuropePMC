#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################

import codecs
from numpy import asarray
from numpy import argmax
from numpy import zeros
from numpy import random
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras import models

from keras.callbacks import TensorBoard

##################################################    Variables     ###################################################

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

dataset="Dataset23.csv"
embedding_dims=50 # Here 50/100/200/300
epochs=12

result_output="ResultEmbedding"+str(embedding_dims)+"d.csv"
embedding_file='glove.6B.'+str(embedding_dims)+'d.txt'

vocab_size=500
average="macro" # binary | micro | macro | weighted | samples
class_weight={
	0 : 25.,
	1 : 20.,
	2 : 10.,}
	# 3 : 10.}
k_cross_val=4
skf=StratifiedKFold(
	n_splits=k_cross_val,
	random_state=42)
Section_num_str,SubType_num_str,Figure_num_str="Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str,completeCitation,completeCitationEmbedd="PreCitation","Citation","PostCitation","CompleteCitation","completeCitationEmbedd"
featuresList=[
	Section_num_str,
	SubType_num_str,
	Figure_num_str,
	'Categories_num']
target_names=[
	"Background",
	# "Compare",
	"Creation",
	"Use"]
# Lemmatizer & Stemmer
lemmatizer=WordNetLemmatizer()
stemmer=SnowballStemmer('english',ignore_stopwords=True)

##################################################    Class     ###################################################

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
	tokens=word_tokenize(doc)
	tokens=[lemma_word(t) for t in tokens]
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
	tokens=word_tokenize(doc)
	tokens=[stem_word(t) for t in tokens]
	return tokens
#
def tokenizer(doc):
	""" This function take as args a doc that could be tokenize.

	Args :
		- doc : (str) a string that can be tokenize by the word_tokenize of nltk library
	
	Return : 
		- tokens : (list) a list of tokens where each token corresponds to a word
	"""
	tokens=word_tokenize(doc)
	return tokens
#
###################################################    Main     ###################################################
#
data=read_csv(dataset,header=0,sep=";")
#
data[completeCitation]=data[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]), axis=1)
#
data["Categories_num"]=data.Categories.map({
	"Background":0,
	"Creation":1,# "Compare":1,
	# "Creation":2,
	"Use":2})# "Use":3})
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
###########################################################################################
lemma_citation=[]
stem_citation=[]
for citation in data[completeCitation]:
	lemma_citation.append(" ".join(lemma_tokenizer(citation)))
	stem_citation.append(" ".join(stem_tokenizer(citation)))

data["lemma_citation"]=lemma_citation
data["stem_citation"]=stem_citation

approaches=[data[completeCitation],data["lemma_citation"],data["stem_citation"]]

output_file=codecs.open(
	filename=result_output,
	mode='w',
	encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tLoss\tf1-scoreCV\tPrecisionCV\tRecallCV\tAccuracyCV\tLossCV\tCombination\tToken\tLemma\tStem\tTime\n")
for approach in approaches:

	tokenizer=Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(approach)
	tmp=tokenizer.texts_to_sequences(approach)

	word_index=tokenizer.word_index

	max_len=len(max(tmp, key=len))

	tmp=DataFrame(pad_sequences(
		sequences=tmp,
		maxlen=max_len, 
		padding='post'))

	data=concat(
		objs=[data[featuresList],tmp],
		axis=1)
	tmp=None

	X=data.drop(['Categories_num'],axis=1)
	y=data.Categories_num
	### !!! ###
	X_to_train,X_val,y_to_train,y_val=train_test_split(X,y,random_state=42)
	X_val=[X_val.iloc[:, 3:],X_val.iloc[:, :3]]
	### !!! ###
	start=time.time()
	f1_score_list,precision_list,recall_list,accuracy_list=[],[],[],[]
	val_acc_list,val_loss_list=[],[]
	control=0
	for train_index,test_index in skf.split(X_to_train,y_to_train):
    	
		NAME="Embedding-"+str(embedding_dims)+"D-epochs"+str(epochs)+"-"+str(approach.name)+str(control)+"-{}".format(int(time.time()))
		tensorboard=TensorBoard(log_dir='./logsEmbedding/{}'.format(NAME))

		X_train, X_test=[X_to_train.iloc[train_index,], X_to_train.iloc[test_index,]] 
		y_train, y_test=[y_to_train.iloc[train_index,], y_to_train.iloc[test_index,]]

		X_train=[X_train.iloc[:, 3:],X_train.iloc[:, :3]] #seq_features,other_features
		X_test=[X_test.iloc[:, 3:], X_test.iloc[:, :3]] #seq_features,other_features

		embeddings_index={}
		f=codecs.open(
			filename=embedding_file,
			mode='r',
			encoding='utf-8')
		for line in f:
			values=line.split()
			word=values[0]
			coefs=asarray(
				a=values[1:],
				dtype='float32')
			embeddings_index[word]=coefs
		f.close()

		not_in_embedding=0
		# embedding_matrix=zeros((len(word_index),embedding_dims))
		embedding_matrix=random.uniform(-0.5,0.5,(len(word_index),embedding_dims))
		for word,i in word_index.items():
			embedding_vector=embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i]=embedding_vector
			else:
				not_in_embedding+=1
		print(not_in_embedding,"/",len(word_index))
		###
		input_layer=layers.Input(
			shape=(X_train[0].shape[1],))

		embedding=layers.Embedding(
			input_dim=len(word_index),
			output_dim=embedding_dims,
			weights=[embedding_matrix],
			input_length=X_train[0].shape[1],
			trainable=False)(input_layer)

		seq_features=layers.Flatten()(embedding)

		other_features=layers.Input(
			shape=(3,))

		model=layers.Concatenate(
			axis=1)([seq_features,other_features])

		model=layers.Dropout(
			rate=.4)(model)

		model=layers.Dense(
			units=len(target_names),
			activation='softmax')(model)

		model=models.Model([input_layer,other_features],model)

		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=['accuracy'])

		model.fit(
			X_train,
			y_train,
			epochs=epochs,
			batch_size=20,
			class_weight=class_weight,
			validation_data=(X_test,y_test),
			callbacks=[tensorboard])

		val_loss,val_acc=model.evaluate(X_test,y_test)
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)

		result=model.predict(X_test)

		y_pred_class=[]
		for sample in result:
			y_pred_class.append(argmax(sample))

		f1_score=metrics.f1_score(y_test,y_pred_class,average=average)*100
		f1_score_list.append(f1_score)
		precision=metrics.precision_score(y_test,y_pred_class,average=average)*100
		precision_list.append(precision)
		recall=metrics.recall_score(y_test,y_pred_class,average=average)*100
		recall_list.append(recall)
		accuracy=metrics.accuracy_score(y_test,y_pred_class)*100
		accuracy_list.append(accuracy)
		control+=1
	end=time.time()

	fold=0
	f1_score_mean,precision_mean,recall_mean,accuracy_mean=0,0,0,0
	val_acc_mean,val_loss_mean=0,0
	while fold < len(f1_score_list):
		f1_score_mean+=f1_score_list[fold]
		precision_mean+=precision_list[fold]
		recall_mean+=recall_list[fold]
		accuracy_mean+=accuracy_list[fold]
		val_acc_mean+=val_acc_list[fold]
		val_loss_mean+=val_loss_list[fold]
		fold+=1
	f1_score_mean=round(f1_score_mean/len(f1_score_list),3)
	precision_mean=round(precision_mean/len(precision_list),3)
	recall_mean=round(recall_mean/len(recall_list),3)
	accuracy_mean=round(accuracy_mean/len(accuracy_list),3)
	val_acc_mean=round(val_acc_mean/len(val_acc_list),3)
	val_loss_mean=round(val_loss_mean/len(val_loss_list),3)

	### !!! ### VALIDATION SET
	val_loss,val_acc=model.evaluate(X_val,y_val)
	result=model.predict(X_val)
	y_pred_class_val=[]
	for sample in result:
		y_pred_class_val.append(argmax(sample))
	f1_score=round(metrics.f1_score(y_val,y_pred_class_val,average=average)*100,3)
	precision=round(metrics.precision_score(y_val,y_pred_class_val,average=average)*100,3)
	recall=round(metrics.recall_score(y_val,y_pred_class_val,average=average)*100,3)
	accuracy=round(metrics.accuracy_score(y_val,y_pred_class_val)*100,3)
	print(
		"\nVALIDATION SET : \n",
		metrics.classification_report(y_val,y_pred_class_val,target_names=target_names),
		"Method : "+str(approach.name),
		"\nF1_score : "+str(f1_score),
		"\tPrecision : "+str(precision),
		"\tRecall : "+str(recall),
		"\tVal_acc : "+str(round(val_acc*100,3)),
		"\tVal_loss : "+str(round(val_loss,3)),
		"\tTime : "+str(round(end-start,3))+" sec",
		"\n#######################################################")
	### !!! ###

	print(
		"\nCROSS VALIDATION : \n",
		metrics.classification_report(y_test,y_pred_class,target_names=target_names),
		"Method : "+str(approach.name),
		"\nF1_score : "+str(f1_score_mean),
		"\tPrecision : "+str(precision_mean),
		"\tRecall : "+str(recall_mean),
		"\tVal_acc : "+str(val_acc_mean),
		"\tVal_loss : "+str(val_loss_mean),
		"\tTime : "+str(round(end-start,3))+" sec",
		"\n#######################################################")
	
	output_file.write(str(f1_score))
	output_file.write("\t")
	output_file.write(str(precision))
	output_file.write("\t")
	output_file.write(str(recall))
	output_file.write("\t")
	output_file.write(str(round(val_acc*100,3)))
	output_file.write("\t")
	output_file.write(str(round(val_loss,3)))
	output_file.write("\t")
	output_file.write(str(f1_score_mean))
	output_file.write("\t")
	output_file.write(str(precision_mean))
	output_file.write("\t")
	output_file.write(str(recall_mean))
	output_file.write("\t")
	output_file.write(str(val_acc_mean*100))
	output_file.write("\t")
	output_file.write(str(val_loss_mean))
	output_file.write("\t")
	output_file.write(str(approach.name))
	output_file.write("\t")
	if completeCitation in str(approach.name):
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\t")
	if "lemma" in str(approach.name):
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\t")
	if "stem" in str(approach.name):
		output_file.write("True")
	else:
		output_file.write("False")
	output_file.write("\t")
	output_file.write(str(round(end-start,3)))
	output_file.write("\n")
output_file.close()