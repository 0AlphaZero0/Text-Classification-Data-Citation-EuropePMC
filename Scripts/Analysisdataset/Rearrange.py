#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import re # Allows to make regex requests


file=codecs.open("dataset-update.csv","r",encoding="utf-8")
oldFile=file
newFile=codecs.open("dataset-update-rearrange.csv","w",encoding="utf-8")
for line in oldFile.readlines():
    line=line.split("\t")
    accessionNumbers=line[1].split(",")
    for accessionNb in accessionNumbers:
        newFile.write(line[0])#PMCID
        newFile.write("\t")
        newFile.write(accessionNb)#AccessionNb
        newFile.write("\t")
        newFile.write(line[2])#Section
        newFile.write("\t")
        newFile.write(line[3])#SubType
        newFile.write("\t")
        newFile.write(line[4])#Figure
        newFile.write("\t")
        newFile.write(line[5])#Pre-citation
        newFile.write("\t")
        newFile.write(line[6])#Citation
        newFile.write("\t")
        newFile.write(line[7])#Post-citation
newFile.close()

file= codecs.open("dataset-update-rearrange.csv","r",encoding="utf-8")
tmp=file.read()
for char in tmp:
    if ord(char)>128:
        char=''
resultFile=codecs.open("dataset-update.csv","w",encoding="utf-8")
resultFile.write(tmp)
resultFile.close()

os.remove("dataset-update-rearrange.csv")