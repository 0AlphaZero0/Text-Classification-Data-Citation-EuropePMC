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

length=len(os.listdir("./articlesOA"))-1
nonOA,nbArticles=0,0

for file in os.listdir("./articlesOA"):
    if file.endswith("-fulltxt.xml"):
        pass
    elif file.endswith(".xml"):
        nbArticles+=1
        os.system('clear')
        print ("****************  "+str(file)+"  ****************")
        advancement=(nbArticles/length)*100
        print ("Processing..........",str(int(advancement))+"%")
        r=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC"+(str(file).split(".")[0])+"/fullTextXML")
        if "<?properties open_access?>" in r.text:
            file=codecs.open("./articlesOA/"+(str(file).split(".")[0])+"-fulltxt.xml","w",encoding="utf-8")
            file.write(r.text)
            file.close()
        else:
            nonOA+=1
print ("There is "+str(nonOA)+" articles that are non OA.")
print ("There is "+str(length-nonOA)+" articles that are OA.")
print ("DONE")
