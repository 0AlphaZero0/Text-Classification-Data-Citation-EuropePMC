#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
from lxml import etree # Allows to create tree from XML file
import os # Allows to modify some things on the os
import sys
import time # Allows to set some point in the execution time and then calculate the execution time

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script should remove tables from full text XML files in the directory set as argument 1.
In the terminal it display some information like execution time, start and stop.
"""

###################################################    Main     ###################################################
print ("###   2/6 - REMOVE TABLES   ###\n")

start=time.time() # start time
for file in os.listdir(sys.argv[1]): # loop for each file in the directory passed as argument 1
    if file.endswith("-fulltxt.xml"): # use only the fulltxt files
        with codecs.open(sys.argv[1]+"/"+str(file),"r",encoding="utf-8") as old: # open the full text and create a tree with lxml library
            tmpFile=old.read()
            xmltree=etree.fromstring(tmpFile)
            old.close()
        tables=xmltree.findall(".//table") # find all tables in the tree thanks to the node table
        for table in tables: # loop for each table to remove those
            table.getparent().remove(table) 
        with codecs.open(sys.argv[1]+"/"+str(file),"w",encoding="utf-8") as new: # save the full text from the tree without tables
            new.write(etree.tostring(xmltree).decode("utf-8"))
            new.close()
end=time.time() # end time

print ("Duration : "+str(int(end-start))+" sec")
print("\nRemove tables DONE\n")