#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import re # Allows to make regex requests


"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will thanks to the file resultCitations.csv create a dataset of citations, most of the time there is multiple data citation in the same citation so there is a lot of repetition.
An d of course we don't like to had 10 times the same sentence so the script will pack together accession numbers that have the same citation sentence.


"""

###################################################    Main     ###################################################

pmcid=[]
accessionNb=[]
section=[]
subtype=[]
citationBefore=[]
citation=[]
citationAfter=[]

csvfile=codecs.open("resultCitations.csv","r",encoding="utf-8")
for line in csvfile.readlines():
    for char in line:
        if ord(char)>128:
            char=''
    line=line.split("\t")
    if line[2]=="":
        pass
    elif line[5] not in citation:
        pmcid.append(line[0])
        accessionNb.append(line[1])
        section.append(line[2])
        subtype.append(line[3])
        citationBefore.append(line[4])
        citation.append(line[5])
        citationAfter.append(line[6][:-2])
    else:
        indexCitation=citation.index(line[5])
        accessionNb[indexCitation]=accessionNb[indexCitation]+","+line[1]
dataset=codecs.open("./articlesOA/dataset1.csv","w",encoding="utf-8")
indexDataset=0
csvfile.close()
while indexDataset < len(citation):
    dataset.write(pmcid[indexDataset])
    dataset.write("\t")
    dataset.write(accessionNb[indexDataset])
    dataset.write("\t")
    dataset.write(section[indexDataset])
    dataset.write("\t")
    dataset.write(subtype[indexDataset])
    dataset.write("\t")
    dataset.write(citationBefore[indexDataset])
    dataset.write("\t")
    dataset.write(citation[indexDataset])
    dataset.write("\t")
    dataset.write(citationAfter[indexDataset])
    dataset.write("\n")
    indexDataset+=1
dataset.close()
filelen=codecs.open("filelen.csv","w",encoding="utf-8")
for cit in citation:
    filelen.write(str(len(cit)))
    filelen.write("\n")
for cit in citationBefore:
    filelen.write(str(len(cit)))
    filelen.write("\n")
filelen.close()