#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
#import numpy as np # Allows to manipulate the necessary table for sklearn
import os # Allows to modify some things on the os
import random # Allows to use random variables
import re # Allows to make regex requests
#import requests # Allows to make http requests
# import shutil #allows file copy
# import sys # Allow to modify files on the OS
# import time # Allows to make a pause to not overcharge the server
# import webbrowser # Allow to use url to open a webbrowser

#################################    Main     ###################################################

length=len(os.listdir("./articlesOA"))-1
nb_done=0
for file in os.listdir("./articlesOA"):
    if file.endswith(".xml"):
        os.system('clear')
        print(str(file),"*********************************************************************************************************************")
        xml=codecs.open("./articlesOA/"+str(file),"r",encoding="utf-8")
        xml=xml.read()
        body=xml.split("<body>")[1].split("</body>")[0]
        file=codecs.open("./articlesOA/Content/"+(str(file).split(".")[0])+"_body.xml","w",encoding="utf-8")
        file.write(body)
        file.close()
        nb_done+=1
        advancement=(nb_done/length)*100
        print ("Content extraction of articles..........",str(int(advancement))+"%")
print ("DONE")