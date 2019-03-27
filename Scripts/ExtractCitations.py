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
length=(len(os.listdir("./articlesOA"))-1)/2


for file in os.listdir("./articlesOA"):
    if file.endswith("-AccessionNb.xml"):
        accessionNames=[]
        fileAccessionNb=codecs.open("./articlesOA/"+str(file),"r",encoding="utf-8")
        fileAccessionNb=fileAccessionNb.read()
        xmlAccessionNb=etree.fromstring(fileAccessionNb)
        names=xmlAccessionNb.findall(".//name")
        for name in names:
            accessionNames.append(name.text)
        fileSentencized=codecs.open("./Sentencized/XML-cured/"+(str(file).split("-")[0])+".xml","r",encoding="utf-8")
        fileSentencized=fileSentencized.read()
        os.system('clear')
        print (str(file).split("-")[0]+".xml")
        """
        fileSentencized=fileSentencized.split(re.search(r'.+REF.+',fileSentencized).group())[0]+"</text></p></fn></fn-group></back></article>"
        """
        fileSentencized=etree.fromstring(fileSentencized)
        sentences=fileSentencized.findall(".//plain")
        sentencesIndex=0
        while sentencesIndex<len(sentences):
            for accessionNb in accessionNames:
                print (type(sentences[sentencesIndex].text),sentencesIndex)
                if accessionNb in sentences[sentencesIndex].text:
                    print (accessionNames)
                    print (accessionNb)
                    print (sentences[sentencesIndex+2])
                    print (sentences[sentencesIndex-1].text+" "+sentences[sentencesIndex].text+" "+sentences[sentencesIndex+1].text+" "+sentences[sentencesIndex+2].text)
            sentencesIndex+=1