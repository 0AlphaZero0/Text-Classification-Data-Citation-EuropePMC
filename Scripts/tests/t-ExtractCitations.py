#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
#import glob # Allows to go through each file of a dir
from lxml import etree # Allows to manipulate xml file easily
#import numpy as np # Allows to manipulate the necessary table for sklearn
import os # Allows to modify some things on the os
import random # Allows to use random variables
import re # Allows to make regex requests
#import requests # Allows to make http requests
#import shutil #allows file copy
#import sys # Allow to modify files on the OS
#import time # Allows to make a pause to not overcharge the server
#import webbrowser # Allow to use url to open a webbrowser
import xml # Allows to manipulate xml files

#################################    Main     ###################################################
empty=0
empty_articles=[]
for file in os.listdir("./articlesOA/Content"):
    if file.endswith("body.xml"):
        listcitations=[]
        context_line=[]
        citations_context=[]
        print(str(file),"*********************************************************************************************************************")
        xml=codecs.open("./articlesOA/"+str(file),"r",encoding="utf-8")
        article=xml.read()
            #citations_line=re.findall(r"([^.]*\.[^.]*bibr[^.]*\.[^.]*\.[^.]*\.)",line)
        citations=re.findall(r"bib.?[^>]+[<sup>]+([^<]+)</",article) #bib.?[^>]+[<sup>]+([^<]+)</  #bib.?[^>]+>([^<]+)</
        if citations==[]:
            empty+=1
            empty_articles.append(str(file))
            #print (len(citations_line),len(citations))
            # if citations!=[] and citations_line!=[]:
            #     x=0
            #     while x<(len(citations)-1):
            #         citations_context.append([citations[x],citations_line[x]])
            #         x=x+1
        print (citations)
        XML=etree.fromstring(xml)
        print (XML)
        break
print (empty_articles)
print ("There is ",empty,"articles 'without' citations.")


"""
Idées  : 
prendre uniquement les parties de l'article écrite (abstract, intro), etc..
"""