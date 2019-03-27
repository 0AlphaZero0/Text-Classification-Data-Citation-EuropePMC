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