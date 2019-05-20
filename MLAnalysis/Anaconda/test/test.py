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
import re

from sklearn.feature_extraction.text import TfidfVectorizer # Allows transformations of string in number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras import models

from keras.callbacks import TensorBoard
######################################
fileCitation="datasetSentences.txt"
fileLabel="sentiment_labels.txt"
datasetCitation=read_csv(
    filename=fileCitation,
    header=True,
    sep="\t")
datasetLabel=read_csv(
    filename=fileLabel,
    header=True,
    sep="|")
# datasetCitation=codecs.open(
#     filename=fileCitation,
#     mode="r",
#     encoding="utf-8")