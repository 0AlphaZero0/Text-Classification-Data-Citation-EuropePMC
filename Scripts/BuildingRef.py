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
length=(len(os.listdir("./articlesOA/Content"))-1)/2
nb_done=0
for file in os.listdir("./articlesOA/Content"):
    if file.endswith("ref.xml"):
        os.system('clear')
        print(str(file),"*********************************************************************************************************************")
        xml=codecs.open("./articlesOA/Content/"+str(file),"r",encoding="utf-8")
        xml=xml.read()
        list_xml=xml.split("/ref")
        list_xml.pop(-1)
        #print (list_xml[0],"////////////////////////////////////////////",list_xml[1])#have to find label + id
        for ref in list_xml:
            print (ref)
            label=re.search(r'<label>([^<]+)</label>',ref)
            print (label.group(0),"//////////////////")
print ("DONE")