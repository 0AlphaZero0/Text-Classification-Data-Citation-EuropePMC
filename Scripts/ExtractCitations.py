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
        fileAccessTmp=fileAccessionNb.read()
        fileAccessionNb.close()
        xmlAccessionNb=etree.fromstring(fileAccessTmp)
        names=xmlAccessionNb.findall(".//name")
        for name in names:
            if name not in accessionNames:
                accessionNames.append(name.text)
        fileSentencized=codecs.open("./Sentencized/XML-cured/"+(str(file).split("-")[0])+".xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        fileSentencized.close()
        os.system('clear')
        print (str(file).split("-")[0]+".xml")
        fileSentencizedTmp=etree.fromstring(fileSentencizedTmp)
        sentences=fileSentencizedTmp.findall(".//SENT")
        sentencesIndex=0
        while sentencesIndex<len(sentences):
            for accessionNb in accessionNames:
                tmp=sentences[sentencesIndex]
                tmpbefore=sentences[sentencesIndex-1]
                if accessionNb in ''.join(tmp.itertext()):
                    tmpafter=''
                    if sentencesIndex+1<len(sentences):
                        tmpafter=tmpafter+''.join(sentences[sentencesIndex+1].itertext())
                    if sentencesIndex+2<len(sentences):
                        tmpafter=tmpafter+''.join(sentences[sentencesIndex+2].itertext())
                    print ("\n",accessionNb,"|",sentencesIndex)
                    resultString=''.join(tmp.itertext())
                    resultFinal=tmpafter+resultString+tmpafter
                    print (resultFinal)
                # print ("ERROR")
                # print (file)
                # print (accessionNames)
                # print (accessionNb)
                # print (sentencesIndex)
                # print (sentences[sentencesIndex].get("sid"))
                # print (type(sentences[sentencesIndex].text),"sentences")
            sentencesIndex+=1