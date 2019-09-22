#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 23/05/2019
########################
from pandas import read_csv # Allows to load csv with pandas library
from pandas import DataFrame # Allows to use dataframe from pandas library
import time # Allows to set some point in the execution time and then calculate the execution time

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script remove duplicates in a dataset to finally use this one to train a machine learning model.
"""

###################################################    Main     ###################################################

print("###   6/6 - REMOVE DUPLICATES   ###\n")
start=time.time()
input_filename="Result.csv"
output_filename="ResultingDataset.csv"
input_dataset=read_csv(input_filename,header=0,sep="\t")
output_dataset=input_dataset.drop_duplicates(subset='Citation',keep="first") # remove duplicates
output_dataset.to_csv(path_or_buf=output_filename,sep="\t",index=False)
end=time.time()
print ("Duration : "+str(int(end-start))+" sec")
print("\nRemove duplicates DONE\n")