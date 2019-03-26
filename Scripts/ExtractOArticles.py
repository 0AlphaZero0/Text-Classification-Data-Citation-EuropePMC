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

nbArticlesScan,nonOA=0,0

while len(os.listdir("./articlesOA"))<401:
    pmcid=str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))
    rAccessionNb=requests.get("https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=PMC%3A"+pmcid+"&type=Accession%20Numbers&format=XML")
    if "<name>" in rAccessionNb.text:
        rFullTxt=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC"+pmcid+"/fullTextXML")
        if "<?properties open_access?>" in rFullTxt.text:
            fileAccessionNb=codecs.open("./articlesOA/PMC"+pmcid+"-AccessionNb.xml","w",encoding="utf-8")
            fileAccessionNb.write(rAccessionNb.text)
            fileAccessionNb.close()
            fileFullTxt=codecs.open("./articlesOA/PMC"+pmcid+"-fulltxt.xml","w",encoding="utf-8")
            fileFullTxt.write(rFullTxt.text)
            fileFullTxt.close()
            nbArticlesScan+=1
        else:
            nonOA+=1
            nbArticlesScan+=1
    else:
        nonOA+=1
        nbArticlesScan+=1
    os.system('clear')
    advancement=(len(os.listdir("./articlesOA"))/400)*100
    print ("Extraction of articles..........",str(int(advancement))+"%")
print ("There is "+str(nonOA)+" articles that are non OA.")
print ("There is "+str(nbArticlesScan-nonOA)+" articles that are OA.")
print ("There is a total of "+str(nbArticlesScan)+" articles that were 'scanned'.")
print ("\nDONE\n")

