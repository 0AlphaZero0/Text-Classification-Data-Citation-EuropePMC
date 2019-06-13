#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
from lxml import etree
import os # Allows to modify some things on the os
import random # Allows to use random variables
import re
import requests # Allows to make http requests
import sys
import time


###################################################    Main     ###################################################
start=time.time()
print ("###   2/6 - REMOVE TABLES   ###\n")
for file in os.listdir(sys.argv[1]):
    if file.endswith("-fulltxt.xml"):
        oldFile=codecs.open(sys.argv[1]+"/"+str(file),"r",encoding="utf-8")
        tmpFile=oldFile.read()
        xmltree=etree.fromstring(tmpFile)
        oldFile.close()
        tables=xmltree.findall(".//table")
        for table in tables:
            table.getparent().remove(table)
        # tables=xmltree.findall(".//table-wrap")
        # for table in tables:
        #     table.getparent().remove(table)
        newFile=codecs.open(sys.argv[1]+"/"+str(file),"w",encoding="utf-8")
        newFile.write(etree.tostring(xmltree).decode("utf-8"))
        newFile.close()
end=time.time()
print ("Duration : "+str(int(end-start))+" sec")
print("\nRemove tables DONE\n")