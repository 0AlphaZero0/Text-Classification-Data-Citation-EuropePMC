#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 23/05/2019
########################
import codecs
import os
from pandas import read_csv
from pandas import DataFrame
import time

###################################################    Main     ###################################################

print("###   6/6 - REMOVE DUPLICATES   ###\n")
start=time.time()
input_filename="Result.csv"
output_filename="Dataset.csv"
input_dataset=read_csv(input_filename,header=0,sep="\t")
output_dataset=input_dataset.drop_duplicates(subset='Citation',keep="first")
output_dataset.to_csv(path_or_buf=output_filename,sep="\t",index=False)
end=time.time()
print ("Duration : "+str(int(end-start))+" sec")
print("\nRemove duplicates DONE\n")