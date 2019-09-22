#!/bin/bash
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################

directory="./articlesOA"
numberofpaper=15
python t1-ExtractOArticles.py $directory $numberofpaper
python t2-RemoveTables.py $directory
bash t3-Sentencizer.sh $directory
python t4-RepareSplitedXML.py $directory
python t5-ExtractCitations.py $directory
python t6-removeDuplicates.py $directory
