#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 01/04/2019
########################
import codecs # Allows to load a file containing UTF-8 characters
from difflib import SequenceMatcher # Allows to give string similarity
from lxml import etree # Allows to manipulate xml file easily
import os # Allows to modify some things on the os
import re # Allows to make regex requests
import xml # Allows to manipulate xml files

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will extract citations from XML sentencized files that are located in the articlesOA directory. Then it will save those citations in a "pre-dataset" called resultCitations.csv.
It 's organized like this :
-------------------------------------------------------------------------------------------
PMCID     AccesionNb     Section     SubType     Pre-Citation     Citation     PostCitation
___________________________________________________________________________________________
Indeed the context of citation is really important so there is a need to extract those.
In the fist time the script will load a file that end like "-AccessionNb.xml" it contain the Accession numbers of a specific paper (they are find thanks to the lxml etree module : .findall(".//name")).
Then the script will load the corresponding sentencized XML file. If the accession number is in the sentence then the previous sentence and the two after are saved in the file.
"""


###################################################    Main     ###################################################
length=(len(os.listdir("./articlesOA"))-1)/2
minlen=25
maxlen=500

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
        print (file)
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
        fileSentencized=codecs.open("./articlesOA/"+(str(file).split("-")[0])+"-sentencized.xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        fileSentencized.close()
        fileSentencizedTmp=etree.fromstring(fileSentencizedTmp)
        # Remove Tables
        for table in fileSentencizedTmp.xpath("//SecTag"):
            if table.attrib["type"]=="TABLE":
                table.getparent().remove(table)
        sentences=fileSentencizedTmp.findall(".//SENT")
        sentencesIndex=0
        while sentencesIndex<len(sentences):
            for accessionNb in accessionNames:
                section=''
                tmp=sentences[sentencesIndex]
                for secTagCitation in tmp.iterancestors("SecTag"):
                    secTagCitation=secTagCitation.get("type")
                for secTagCitationBefore in sentences[sentencesIndex-1].iterancestors("SecTag"):
                    secTagCitationBefore=secTagCitationBefore.get("type")
                if secTagCitation==secTagCitationBefore and minlen<len(''.join(sentences[sentencesIndex-1].itertext()))<maxlen:
                    tmpbefore=''.join(sentences[sentencesIndex-1].itertext())
                if accessionNb in ''.join(tmp.itertext()) and minlen<len(''.join(tmp.itertext()))<maxlen:
                    tmpafter=''
                    for preCitPost in preCitPosts:# this loop is made to extract section type & subtype
                        if preCitPost[0] in ''.join(tmp.itertext()):
                            section=preCitPost[1]
                            if "(" in section:
                                section=section.split(" (")[0]
                                subtype=preCitPost[2]
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
                                    subtype=preCitPost[2]
                    if sentencesIndex+1<len(sentences):
                        for secTagCitation1 in sentences[sentencesIndex+1].iterancestors("SecTag"):
                            secTagCitation1=secTagCitation1.get("type")
                        if secTagCitation==secTagCitation1 and minlen<len(''.join(sentences[sentencesIndex+1].itertext()))<maxlen:
                            tmpafter=tmpafter+''.join(sentences[sentencesIndex+1].itertext())
                            if sentencesIndex+2<len(sentences):
                                for secTagCitation2 in sentences[sentencesIndex+2].iterancestors("SecTag"):
                                    secTagCitation2=secTagCitation2.get("type")
                                if secTagCitation==secTagCitation2 and minlen<len(''.join(sentences[sentencesIndex+2].itertext()))<maxlen:
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