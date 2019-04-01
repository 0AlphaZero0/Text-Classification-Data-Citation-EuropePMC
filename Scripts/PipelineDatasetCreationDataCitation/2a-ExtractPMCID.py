#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will extract PMCID of all papers that were previously extracted and then had then to a csv file that is needed for the sentences splitter.
"""

#################################    Main     ###################################################
pmcidlist=codecs.open("pmcidList.csv","w",encoding="utf-8")
for file in os.listdir("./articlesOA"):
    if file.endswith("-fulltxt.xml"):
        pmcidlist.write(str(file).split("-")[0])
        pmcidlist.write("\n")
pmcidlist.close()
        