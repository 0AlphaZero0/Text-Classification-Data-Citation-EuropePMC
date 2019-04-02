#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import re # Allows to make regex requests

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will look at the mistakes made by the sentence splitter, indeed sometimes, the sentence splitter just split sentences in the wrong places so it need to be fix.
So here thanks to regular expression it will fix some mistakes made by the splitter in the xml files containing the sentences.

"""

###################################################    Main     ###################################################

for file in os.listdir("./articlesOA"):
    if file.endswith("-sentencized.xml"):
        oldFile=codecs.open("./articlesOA/"+str(file),"r",encoding="utf-8")
        tmpFile=oldFile.read()
        oldFile.close()
        matchesEtAl=re.findall(r'\set\sal\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesSurname=re.findall(r'\s[^h\d%;]\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesCa=re.findall(r'\sca\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesApprox=re.findall(r'\sapprox\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesRef=re.findall(r'\(ref\.\s(</plain></SENT>\n<[^>]+pm=\"\.\"><plain>)',tmpFile)
        matchesTable=re.findall(r'(<table-wrap[\s\S]+</table-wrap>)',tmpFile)
        for match in matchesEtAl:
            tmpFile=tmpFile.replace(match,'')
        for match in matchesSurname:
            tmpFile=tmpFile.replace(match,'')
        for match in matchesCa:
            tmpFile=tmpFile.replace(match,'')
        for match in matchesApprox:
            tmpFile=tmpFile.replace(match,'')
        for match in matchesRef:
            tmpFile=tmpFile.replace(match,'')
        for match in matchesTable:
            tmpFile=tmpFile.replace(match,'')
        newFile=codecs.open("./articlesOA/"+str(file),"w",encoding="utf-8")
        newFile.write(tmpFile)
        newFile.close()
