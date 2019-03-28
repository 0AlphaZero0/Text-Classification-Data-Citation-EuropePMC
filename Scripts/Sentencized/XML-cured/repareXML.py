#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
#import numpy as np # Allows to manipulate the necessary table for sklearn
import os # Allows to modify some things on the os
import random # Allows to use random variables
import re # Allows to make regex requests
import requests # Allows to make http requests
# import shutil #allows file copy
# import sys # Allow to modify files on the OS
import time # Allows to make a pause to not overcharge the server
# import webbrowser # Allow to use url to open a webbrowser

#################################    Main     ###################################################

for file in os.listdir("./"):
    if file.endswith(".xml"):
        oldFile=codecs.open(file,"r",encoding="utf-8")
        tmpFile=oldFile.read()
        oldFile.close()
        matchesEtAl=re.findall(r'\set\sal\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesSurname=re.findall(r'\s[^h\d%;]\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesCa=re.findall(r'\sca\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesApprox=re.findall(r'\sapprox\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesRef=re.findall(r'\(ref\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesEtAl:
            #print ("match1")
            tmpFile=tmpFile.replace(match,'')
        for match in matchesSurname:
            #print ("match2")
            tmpFile=tmpFile.replace(match,'')
        for match in matchesCa:
            #print ("match3")
            tmpFile=tmpFile.replace(match,'')
        for match in matchesApprox:
            #print ("match4")
            tmpFile=tmpFile.replace(match,'')
        for match in matchesRef:
            #print ("match5")
            tmpFile=tmpFile.replace(match,'')
        newFile=codecs.open(file,"w",encoding="utf-8")
        newFile.write(tmpFile)
        newFile.close()
    else:
        print (file)
