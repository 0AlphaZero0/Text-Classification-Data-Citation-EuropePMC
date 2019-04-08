#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
import os # Allows to modify some things on the os

file= codecs.open("dataset-test-rearrange.csv","r",encoding="utf-8")
tmp=file.read()
for char in tmp:
    if ord(char)>128:
        char=''
resultFile=codecs.open("dataset-fixed.csv","w",encoding="utf-8")
resultFile.write(tmp)
