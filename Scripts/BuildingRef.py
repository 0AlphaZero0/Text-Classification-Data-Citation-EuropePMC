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
articles_ref_list=[]
not_journal=0
nb_citations=0
for file in os.listdir("./articlesOA/Content"):
    if file.endswith("ref.xml"):
        TMP=[]
        TMP.append(str(file)[:-8])
        os.system('clear')
        print(str(file),"*********************************************************************************************************************")
        xml=codecs.open("./articlesOA/Content/"+str(file),"r",encoding="utf-8")
        xml=xml.read()
        list_xml=xml.split("/ref")
        list_xml.pop(-1)
        #print (list_xml[0],"////////////////////////////////////////////",list_xml[1])#have to find label + id
        allRefForAPaper=[]
        for ref in list_xml:
            nb_citations+=1
            #print (ref,'\n')
            # label to find in papers's bodies
            try:
                label=re.search(r'<label>([^<]+)</label>',ref)
                label=label.group(1)
            except AttributeError:
                label=re.search(r'ref id=\"([^\"]+)\"',ref)#group 0 = entire match, group 1 = first match
                label=label.group(1)
            # ADD anything interesting in the ref
            citation_type=re.search(r'publication-type=\"([^\"]+)\"',ref)
            if citation_type.group(1)=="journal":
                pmid=re.search(r'pmid\">([^<]+)<',ref)
                try:
                    pmid=pmid.group(1)#####!!!!!!!!! Sometimes articles doesn't have a pmid in xml but it actually exists
                except AttributeError:
                    pmid=""
                    
            else:
                pmid=""
                not_journal+=1
            allRefForAPaper.append([label,pmid])
        TMP.append(allRefForAPaper)
        articles_ref_list.append(TMP)
print (articles_ref_list)

print ("There is ",not_journal," citations that aren't journal citation.")
print ("On ",nb_citations," citations....",(not_journal/nb_citations)*100," are not journal citations.")
print ("DONE")