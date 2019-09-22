#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import random # Allows to use random variables
import requests # Allows to make http requests
import sys # Allows access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import time # Allows to set some point in the execution time and then calculate the execution time

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script should extract two type of information from EuropePMC, first Full-text papers in XML format, anf then an XML that contain all Accession Number of this paper.
Papers are extract randomly thanks to "random" library, they are identify by a PMCID that looks like PMCXXXXXXX where each X can take a value from 0 to 9.
Then thanks to PMCID created a request is made to see if the extracted papers contain a data citation,
thanks to the "<name>" tag (in the XML file containing data citations) that indicate there is at least one data citation.
Then a second request is made to extract the fullText file in XML format and to be sure that it's an Open Access paper, it look at the tag "<?properties open_access?>".
Downloaded files are store in a directory named articlesOA and this script will make those requests until there is 400 files in the directory or 200 open access papers.
In the terminal it display some information like the advancement of the extraction, at the number of papers that were requested and the number of open access one and non open access ones.
"""



###################################################    Main     ###################################################

nbArticlesScan=0 # Number of pmcid created in 1 run
directory=sys.argv[1] # Directory in which articles are saved.
nb_of_article=sys.argv[2] # Number of papers containing accession numbers and are open access to download
listPmcidScannedNonOA=[] # List of pmcid that are not open access or don't contain any accession numbers
listPmcidScannedOA=[] # List of pmcid open access and containing accession numbers
with open(file="pmcid_scanned_nonOA.txt",mode="r",encoding="utf-8") as f: # load pmcid non open access or not containing accession numbers in a list
    for line in f.readlines():
        listPmcidScannedNonOA.append(line.split("\n")[0])
    f.close()
with open(file="pmcid_scanned.txt",mode="r",encoding="utf-8") as f: # load pmcid open access or containing accession numbers in a list
    for line in f.readlines():
        listPmcidScannedOA.append(line.split("\n")[0])
    f.close()

start=time.time() # start time
while (len(listPmcidScannedOA))<(int(nb_of_article)): # loop to check open access and accession numbers in papers corresponding to random pmcids and then save corresponding files
    pmcid=str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9)) # generate a random pmcid
    if pmcid in listPmcidScannedNonOA or pmcid in listPmcidScannedOA: # check if the random pmcid has already been 'scanned'
        pass
    else:
        nbArticlesScan+=1
        rAccessionNb=requests.get("https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=PMC%3A"+pmcid+"&type=Accession%20Numbers&format=XML") # request Annotations API from EPMC to return the XML file corresponding to previously generated pmcid
        if "<name>" in rAccessionNb.text: # check if there is at least one accession number in the resulting XML
            rFullTxt=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC"+pmcid+"/fullTextXML") # request the complete XML file corresponding to the full text of the previously generated pmcid 
            if "<?properties open_access?>" in rFullTxt.text: # check if the paper is open access
                listPmcidScannedOA.append(pmcid)
                with codecs.open(directory+"/PMC"+pmcid+"-AccessionNb.xml","w",encoding="utf-8") as fAccNb:# save the XML containing the result of Annotation API in a file corresponding to pmcid-AccessionNb.xml
                    fAccNb.write(rAccessionNb.text)
                    fAccNb.close()
                with codecs.open(directory+"/PMC"+pmcid+"-fulltxt.xml","w",encoding="utf-8") as fFulTxt:# save the XML containing the full text in a file corresponding to pmcid-fulltxt.xml
                    fFulTxt.write(rFullTxt.text)
                    fFulTxt.close()
            else:
                listPmcidScannedNonOA.append(pmcid)
        else:
            listPmcidScannedNonOA.append(pmcid)
    
    os.system('clear')
    print ("###   1/6 - EXTRACTION ARTICLES   ###\n")
    advancement=((len(listPmcidScannedOA))/(int(nb_of_article)))*100
    print ("Extraction of documents..........",str(int(advancement))+"%")
end=time.time() # end time

with open(file="pmcid_scanned_nonOA.txt",mode="w",encoding="utf-8") as f: # save in a file list of pmcid already scanned non Open access or not containing accession numbers
    for pmcid in listPmcidScannedNonOA:
        f.write(pmcid)
        f.write("\n")
    f.close()
with open(file="pmcid_scanned.txt",mode="w",encoding="utf-8") as f: # save in a file list of pmcid already scanned Open access and containing accession numbers
    for pmcid in listPmcidScannedOA:
        f.write(pmcid)
        f.write("\n")
    f.close()

# Print a list of informations in the terminal
os.system('clear')
print ("###   1/6 - EXTRACTION ARTICLES   ###\n")
print ("Duration : "+str(int(end-start))+" sec") # execution time
print ("There is "+str(len(listPmcidScannedNonOA))+" documents that are non eligible.") # documents not open access or with 0 accession numbers
print ("There is "+str(int(len(listPmcidScannedOA)))+" documents that are OA and contains accession numbers.") # numbers of pmcid open access and containing accession numbers
print ("There is a total of "+str(int(nbArticlesScan))+" documents that were 'scanned' in this step.") # pmcid generated during this run
print ("There is a total of "+str(int(len(listPmcidScannedNonOA)+len(listPmcidScannedOA)))+" documents that were 'scanned' for all steps.") # pmcid generated for all runs
print ("\nExtraction articles DONE\n")