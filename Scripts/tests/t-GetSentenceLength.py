#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
from difflib import SequenceMatcher # Allows to give string similarity
from lxml import etree # Allows to manipulate xml file easily
import os # Allows to modify some things on the os
import re # Allows to make regex requests
import xml # Allows to manipulate xml files


###################################################    Main     ###################################################

filelength=codecs.open("AllLength.csv","w",encoding="utf-8")
for file in os.listdir("./"):
    if file.endswith(".xml"):
        fileReadTmp=codecs.open("./"+str(file),"r",encoding="utf-8")
        fileRead=fileReadTmp.read()
        fileReadTmp.close()
        Tree=etree.fromstring(fileRead)
        lengths=Tree.findall(".//SENT")
        for sent in lengths:
            length=len(''.join(sent.itertext()))
            filelength.write(str(length))
            filelength.write("\n")
filelength.close()
