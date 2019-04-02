#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
#01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import re # Allows to make regex requests

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

The pre-existing tool that sentencize XML files result in an unique file, but for the extraction it's necesseray to have multpile files, one for each paper, so thanks to the tag "<!DOCTYPE",
the file patch.xml is splitted in X files named thanks to their PMCID, that are find through regular expressions. Files are stored in the articleOA directory.
"""


#################################    Main     ###################################################

patchXml=codecs.open("patch.xml","r",encoding="utf-8")
patchXml=patchXml.read()
patchXml=patchXml.split("<!DOCTYPE")
patchXml.pop(0)
for xmlFile in patchXml:
    xmlFile="<!DOCTYPE"+xmlFile
    pmcid=re.search(r'pmcid\">([0-9]+)',xmlFile).group(1)
    pmcidFile=codecs.open("./XML-sentencized/PMC"+pmcid+".xml","w",encoding="utf-8")
    pmcidFile.write(xmlFile)
    pmcidFile.close()