#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import re # Allows to make regex requests
import sys # Allows access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import time # Allows to set some point in the execution time and then calculate the execution time

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will look at the mistakes made by the sentence splitter, indeed sometimes, the sentence splitter just split sentences in the wrong places so it need to be fix.
So here thanks to regular expressions it will fix some mistakes made by the splitter in the xml files containing the sentences.

"""

###################################################    Main     ###################################################
directory=sys.argv[1]
print ("###   4/6 - REPARE SPLITED .XML   ###\n")

start=time.time() # start time
for file in os.listdir(directory): # check 
    if file.endswith("-sentencized.xml"): # only work on sentencized files
        # file loading
        oldFile=codecs.open(directory+"/"+str(file),"r",encoding="utf-8")
        tmpFile=oldFile.read()
        oldFile.close()
        # fix "et al." mistake
        matchesEtAl=re.findall(r'\set\sal\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesEtAl:
            tmpFile=tmpFile.replace(match,'')
        # fix single letter mistake
        matchesSurname=re.findall(r'\s[^h\d%;]\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesSurname:
            tmpFile=tmpFile.replace(match,'')
        # fix "ca." mistake
        matchesCa=re.findall(r'\sca\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesCa:
            tmpFile=tmpFile.replace(match,'')
        # fix "approx." mistake
        matchesApprox=re.findall(r'\sapprox\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesApprox:
            tmpFile=tmpFile.replace(match,'')
        # fix "ref." mistake
        matchesRef=re.findall(r'\(ref\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesRef:
            tmpFile=tmpFile.replace(match,'')
        # fix "ver." mistake
        matchesVer=re.findall(r'ver\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        for match in matchesVer:
            tmpFile=tmpFile.replace(match,'')
        # save in the file deleting previous one
        newFile=codecs.open(directory+"/"+str(file),"w",encoding="utf-8")
        newFile.write(tmpFile)
        newFile.close()
end=time.time() # end time

print ("Duration : "+str(int(end-start))+" sec")
print("\nRepare splited .xml DONE\n")