#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os
import random # Allows to use random variables
import requests # Allows to make http requests
import sys
import time


file=codecs.open(
    "Result.csv",
    "r",
    encoding="utf-8")

listpmcid=[]

for lines in file.readlines():
    pmcid=str(lines.split("\t")[0])
    if pmcid in listpmcid:
        pass
    else:
        listpmcid.append(str(pmcid))

print(len(listpmcid))