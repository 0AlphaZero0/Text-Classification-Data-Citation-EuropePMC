#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
from difflib import SequenceMatcher # Allows to give string similarity
from lxml import etree # Allows to manipulate xml file easily
import os # Allows to modify some things on the os
import re # Allows to make regex requests
import xml # Allows to manipulate xml files




###################################################    Main     ###################################################
length=(len(os.listdir("./articlesOA"))-1)/2

resultFile=codecs.open("resultCitations.csv","w",encoding="utf-8")
resultFile.write("PMCID")
resultFile.write("\t")
resultFile.write("AccessionNb")
resultFile.write("\t")
resultFile.write("Section")
resultFile.write("\t")
resultFile.write("SubType")
resultFile.write("\t")
resultFile.write("Pre-citation")
resultFile.write("\t")
resultFile.write("Citation")
resultFile.write("\t")
resultFile.write("Post-citation")
resultFile.write("\n")
for file in os.listdir("./articlesOA"):
    if file.endswith("-AccessionNb.xml"):
        accessionNames=[]
        fileAccessionNb=codecs.open("./articlesOA/"+str(file),"r",encoding="utf-8")
        fileAccessTmp=fileAccessionNb.read()
        fileAccessionNb.close()
        xmlAccessionNb=etree.fromstring(fileAccessTmp)
        ### Extract Accession number
        names=xmlAccessionNb.findall(".//name")
        for name in names:
            if name.text not in accessionNames:
                accessionNames.append(name.text)
        ### Extract Section type
        prefixes=xmlAccessionNb.findall(".//prefix")
        postfixes=xmlAccessionNb.findall(".//postfix")
        sections=xmlAccessionNb.findall(".//section")
        subtypes=xmlAccessionNb.findall(".//subType")
        preCitPosts=[]
        index=0
        while index<len(names):
            preCitPosts.append([''.join(prefixes[index].itertext())+''.join(names[index].itertext())+''.join(postfixes[index].itertext()),''.join(sections[index].itertext()),''.join(subtypes[index].itertext())])
            index+=1
        fileSentencized=codecs.open("./Sentencized/XML-cured/"+(str(file).split("-")[0])+".xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        fileSentencized.close()
        fileSentencizedTmp=etree.fromstring(fileSentencizedTmp)
        sentences=fileSentencizedTmp.findall(".//SENT")
        sentencesIndex=0
        while sentencesIndex<len(sentences):
            for accessionNb in accessionNames:
                section=''
                tmp=sentences[sentencesIndex]
                tmpbefore=''.join(sentences[sentencesIndex-1].itertext())
                if accessionNb in ''.join(tmp.itertext()):
                    tmpafter=''
                    for preCitPost in preCitPosts:# this loop is made to extract section type & subtype
                        if preCitPost[0] in ''.join(tmp.itertext()):
                            section=preCitPost[1]
                            if "(" in section:
                                section=section.split(" (")[0]
                                subtype=subtypes[2]
                        else:# this is not really accurate maybe should I check later the ratio
                            tmpStrRatio=''.join(tmp.itertext())
                            indexMatch=tmpStrRatio.find(accessionNb)
                            beginning=indexMatch-20
                            if beginning <0:
                                beginning=0
                            ending=indexMatch+len(accessionNb)+20
                            if ending>len(tmpStrRatio)-1:
                                ending=len(tmpStrRatio)
                            tmpStrRatio=tmpStrRatio[beginning:ending]
                            strRatio=SequenceMatcher(None,tmpStrRatio,preCitPost[0])
                            if strRatio.ratio()>0.50: #50% of similarity through those string.
                                section=preCitPost[1]
                                if "(" in section:
                                    section=section.split(" (")[0]
                                    subtype=subtypes[2]
                    if sentencesIndex+1<len(sentences):
                        tmpafter=tmpafter+''.join(sentences[sentencesIndex+1].itertext())
                        if sentencesIndex+2<len(sentences):
                            tmpafter=tmpafter+''.join(sentences[sentencesIndex+2].itertext())
                    resultString=''.join(tmp.itertext())
                    resultFile.write(str(file).split("-")[0])
                    resultFile.write("\t")
                    resultFile.write(accessionNb)
                    resultFile.write("\t")
                    resultFile.write(section)
                    resultFile.write("\t")
                    resultFile.write(subtype)
                    resultFile.write("\t")
                    resultFile.write(tmpbefore)
                    resultFile.write("\t")
                    resultFile.write(resultString)
                    resultFile.write("\t")
                    resultFile.write(tmpafter)
                    resultFile.write("\n")
            sentencesIndex+=1
resultFile.close()
print("DONE")