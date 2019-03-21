#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
import numpy as np # Allows to manipulate the necessary table for sklearn
import os # Allows to modify some things on the os
import random # Allows to use random variables
import re # Allows to make regex requests
import requests # Allows to make http requests
import shutil #allows file copy
import sys # Allow to modify files on the OS
import time # Allows to make a pause to not overcharge the server
import webbrowser # Allow to use url to open a webbrowser

#################################    Main     ###################################################

id_list=[]
nonOA,nb_article=0,1

r = requests.get ("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC/PMC3492754/citations?page=1&pageSize=1000&format=xml")
id_list=re.findall(r"<id>(.[^<]+)</id>",r.text)
id_article_cited=id_list.pop(0)
for article_id in id_list:
    time.sleep(0.2)
    r = requests.get ("https://www.ebi.ac.uk/europepmc/webservices/rest/"+article_id+"/fullTextXML")
    if "<?properties open_access?>" not in r.text or r.text=="":
        nonOA+=1
        pass
    else :
        file=codecs.open("./articlesOA/"+article_id+".xml","w",encoding="utf-8")
        file.write(r.text)
        file.close
    advancement = (nb_article/(len(id_list)))*100
    nb_article+=1
    os.system('clear')
    print (str(int(advancement))+"%")
print ("There is "+str(nonOA)+" articles that are non OA.")
print ("There is "+str(nb_article-nonOA)+" articles that are OA.")
print ("There is "+str(nb_article)+" citations of this article.")
print ("\nDONE\n")
#print (idlist)
#print (len(idlist))