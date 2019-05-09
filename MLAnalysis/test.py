#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################

import codecs
from numpy import concatenate
from numpy import argmax
from pandas import read_csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import normalize
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

##################################################    Variables     ###################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


filename = "Dataset2.csv"
result_output="ResultDLparam.csv"
average="macro" # binary | micro | macro | weighted | samples
class_weight = {
	0 : 15.,
	1 : 50.,
	2 : 15.,
	3 : 10.}
epochs = 5
# input_node = 1280
activation_input_node = 'relu'
node1 = 128
activation_node1 = 'relu'
node2 = 128
activation_node2 = 'relu'
output_node = 4
activation_output_node='softmax'
ngram_range = (1,3)
token,ngram,lemma,stem = "Tokenization","N-gram","Lemmatization","Stemming"
Section_num_str,SubType_num_str,Figure_num_str = "Section_num","SubType_num","Figure_num"
PreCitation_str,Citation_str,PostCitation_str,completeCitation = "PreCitation","Citation","PostCitation","CompleteCitation"
featuresList = [
	Section_num_str,
	SubType_num_str,
	Figure_num_str,
	completeCitation]
target_names = [
	"Background",
	"Compare",
	"Creation",
	"Use"]
extra_features = [
	token,
	ngram,
	lemma,
	stem]

##################################################    Class     ###################################################

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
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
		j = 0
		while j<len(src):
			fn(n-1, src[:j], [src[j]] + got, all)
			j = j+1
		return
	all = []
	i = 0
	while i<len(a):
		fn(i,a,[],all)
		i = i+1
	all.append(a)
	return all #a = [1,2,3,4] print(combinations(a))

###################################################    Main     ###################################################
#
data = read_csv(filename,header = 0,sep = "\t")
#
data[completeCitation] = data[[PreCitation_str,Citation_str,PostCitation_str]].apply(lambda x : '{}{}'.format(x[0],x[1]), axis = 1)
#
data["Categories_num"] = data.Categories.map(
	{"Background":0,
	"Compare":1,
	"Creation":2,
	"Use":3})
#
data[Figure_num_str] = data.Figure.map(
	{True:0,
	False:1})
#
sectionDict = {}
index = 1
for section in data.Section:
	if section not in sectionDict:
		sectionDict[section] = index
		index+=1
data[Section_num_str] = data.Section.map(sectionDict)
#
subTypeDict = {}
index = 1
for subType in data.SubType:
	if subType not in subTypeDict:
		subTypeDict[subType] = index
		index+=1
data[SubType_num_str] = data.SubType.map(subTypeDict)
#
##################################################################
#
X = data[featuresList]
y = data.Categories_num

combinations_list = combinations(extra_features)

output_file=codecs.open(result_output,'w',encoding='utf8')
output_file.write("f1-score\tPrecision\tRecall\tAccuracy\tCross-score\tLoss\tCombination\tToken\tNgram\tLemma\tStem\n")
for combination in combinations_list:
	accuracy_list=[]
	for i in range(5):
		vect_list = [
			[TfidfVectorizer(), completeCitation, token],
			[TfidfVectorizer(ngram_range = ngram_range), completeCitation, ngram],
			[TfidfVectorizer(tokenizer = LemmaTokenizer()), completeCitation, lemma],
			[TfidfVectorizer(analyzer = stemmed_words), completeCitation, stem]]
		#
		stemmer = SnowballStemmer('english',ignore_stopwords = True)
		analyzer = TfidfVectorizer().build_analyzer()
		print(str(i+1)+"/5 runs")
		X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)
		vect_X_train,vect_X_test = [],[]
		vect_tmp=[]
		for vect in vect_list:
			if vect[2] in combination:
				vect_tmp.append(vect[2])
				print(vect[2])
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
		
		X_train_dtm = concatenate(vect_X_train, axis = 1)
		X_train_dtm = normalize(X_train_dtm, axis = 1)

		X_test_dtm = concatenate(vect_X_test, axis = 1)
		X_test_dtm = normalize(X_test_dtm, axis = 1)

		print (X_train_dtm[0].shape[1])
		model = Sequential([
			Dense(1280, activation = activation_input_node, input_dim=X_train_dtm[0].shape[1]),
			Dense(node1, activation = activation_node1),
			Dense(node2, activation = activation_node2),
			Dense(output_node, activation = activation_output_node),
			])

		model.compile(
			optimizer="adam",
			loss="sparse_categorical_crossentropy",
			metrics=['accuracy'])

		model.fit(
			X_train_dtm,
			y_train,
			epochs = epochs,
			batch_size = 20,
			class_weight = class_weight)

		val_loss, val_acc = model.evaluate(X_test_dtm, y_test)

		result = model.predict(X_test_dtm)
		
		y_pred=[]
		for sample in result:
			y_pred.append(argmax(sample))
		
		f1_score = round(metrics.f1_score(y_test, y_pred, average = average)*100,3)
		precision = round(metrics.precision_score(y_test, y_pred, average = average)*100,3)
		recall = round(metrics.recall_score(y_test, y_pred, average = average)*100,3)
		accuracy_list.append(val_acc)

	accuracy_mean = 0
	for accuracy in accuracy_list:
		accuracy_mean = float(accuracy_mean) + float(accuracy)
	accuracy_mean = accuracy_mean/len(accuracy_list)
	print(
		metrics.classification_report(y_test,y_pred,target_names = target_names),
		"Cross score : "+str(round(accuracy_mean*100,3)),
		"Accuracy score : "+str(round(metrics.accuracy_score(y_test,y_pred)*100,3)),
		"\tF1_score : "+str(f1_score),
		"\tPrecision : "+str(precision),
		"\tRecall : "+str(recall),
		"\n#######################################################")

	output_file.write(str(f1_score))
	output_file.write("\t")
	output_file.write(str(precision))
	output_file.write("\t")
	output_file.write(str(recall))
	output_file.write("\t")
	output_file.write(str(val_acc*100))
	output_file.write("\t")
	output_file.write(str(round(accuracy_mean*100,3)))
	output_file.write("\t")
	output_file.write(str(val_loss))
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
	output_file.write("\n")
		