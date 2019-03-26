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
pmcidlist=codecs.open("pmcidList.csv","w",encoding="utf-8")
for file in os.listdir("./articlesOA"):
    if file.endswith("-fulltxt.xml"):
        pmcidlist.write(str(file).split("-")[0])
        pmcidlist.write("\n")
pmcidlist.close()
        