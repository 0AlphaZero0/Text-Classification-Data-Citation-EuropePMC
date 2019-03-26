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
        for item in xmlAccessionNb:
            for annotations in item:
                if annotations.tag=="annotations":
                    for annotation in annotations:
                        for tags in annotation:
                            if tags.tag=="tags":
                                for tag in tags:
                                    for name in tag:
                                        if name.tag=="name":
                                            accessionNames.append(name.text)
        fileFullText=codecs.open("./articlesOA/"+(str(file).split("-")[0])+"-fulltxt.xml","r",encoding="utf-8")
        fileFullText=fileFullText.read()
        for accessionNb in accessionNames:
            