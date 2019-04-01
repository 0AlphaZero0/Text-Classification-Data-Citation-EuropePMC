#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import re # Allows to make regex requests

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