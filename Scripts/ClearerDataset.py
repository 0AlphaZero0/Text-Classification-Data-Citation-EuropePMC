#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
#import numpy as np # Allows to manipulate the necessary table for sklearn
from lxml import etree # Allows to manipulate xml file easily
import os # Allows to modify some things on the os
import random # Allows to use random variables
import re # Allows to make regex requests
import requests # Allows to make http requests
# import shutil #allows file copy
# import sys # Allow to modify files on the OS
import time # Allows to make a pause to not overcharge the server
# import webbrowser # Allow to use url to open a webbrowser
import xml # Allows to manipulate xml files

#################################    Main     ###################################################

pmcid=[]
accessionNb=[]
citationBefore=[]
citation=[]
citationAfter=[]

csvfile=codecs.open("resultCitations.csv","r",encoding="utf-8")
for line in csvfile.readlines():
    line=line.split("\t")
    if line[3] not in citation:
        pmcid.append(line[0])
        accessionNb.append(line[1])
        citationBefore.append(line[2])
        citation.append(line[3])
        citationAfter.append(line[4])
    else:
        indexCitation=citation.index(line[3])
        accessionNb[indexCitation]=accessionNb[indexCitation]+","+line[1]
dataset=codecs.open("dataset1.csv","w",encoding="utf-8")
indexDataset=0
csvfile.close()
while indexDataset < len(citation):
    dataset.write(pmcid[indexDataset])
    dataset.write("\t")
    dataset.write(accessionNb[indexDataset])
    dataset.write("\t")
    dataset.write(citationBefore[indexDataset])
    dataset.write("\t")
    dataset.write(citation[indexDataset])
    dataset.write("\t")
    dataset.write(citationAfter[indexDataset])
    dataset.write("\n")
    indexDataset+=1
dataset.close()