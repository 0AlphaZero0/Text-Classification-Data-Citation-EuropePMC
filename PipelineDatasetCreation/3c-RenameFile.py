#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will extract PMCID of all papers that were previously extracted and then had then to a csv file that is needed for the sentences splitter.
"""

#################################    Main     ###################################################

for file in os.listdir("./articlesOA"):
    if file.endswith(".xml"):
        if file.endswith("-AccessionNb.xml") or file.endswith("-fulltxt.xml"):
            pass
        else:
            os.rename("./articlesOA/"+str(file),"./articlesOA/"+str(file).split(".")[0]+"-sentencized.xml")