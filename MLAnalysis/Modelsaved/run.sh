#!/bin/sh
directory="./articlesOA"
batchofpaper=1000
result=`python t0-checknbpapers.py`
echo "$result papers containing citations have been extracted and predicted"
for ((i=$result;i<10000;i=$result))
    do
        python t1-ExtractOArticles.py $directory $batchofpaper
        python t2-RemoveTables.py $directory
        bash t3-Sentencizer.sh $directory
        python t4-RepareSplitedXML.py $directory
        python t5-ExtractCitations.py $directory
        rm ./articlesOA/*
        python t6-Predict.py
        python t7-CategorizationByPaper.py
        result=`python t0-checknbpapers.py`
        echo "$result papers containing citations have been extracted and predicted"
    done