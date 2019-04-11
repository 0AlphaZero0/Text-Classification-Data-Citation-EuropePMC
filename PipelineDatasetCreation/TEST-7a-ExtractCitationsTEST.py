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
#################################################    Function     #################################################

def save():
    pass

###################################################    Main     ###################################################
length=(len(os.listdir("./articlesOA"))-1)/2
minlen=25
maxlen=500

outputmistakefile=codecs.open("mistakefile.csv","w",encoding="utf-8")

resultFile=codecs.open("resultCitations.csv","w",encoding="utf-8")
resultFile.write("PMCID")
resultFile.write("\t")
resultFile.write("AccessionNb")
resultFile.write("\t")
resultFile.write("Section")
resultFile.write("\t")
resultFile.write("SubType")
resultFile.write("\t")
resultFile.write("Figure")
resultFile.write("\t")
resultFile.write("Pre-citation")
resultFile.write("\t")
resultFile.write("Citation")
resultFile.write("\t")
resultFile.write("Post-citation")
resultFile.write("\n")

numberOfExtracted=0
numberOfAnnotations=0
threshold=0.85

# for a PMCID file sentencized :
for file in os.listdir("./articlesOA"):
    # Check first the PMCID-AccessionNb.xml file
    if file.endswith("-AccessionNb.xml"):
        print (file)
        accessionNames=[]
        fileAccessionNb=codecs.open("./articlesOA/"+str(file),"r",encoding="utf-8")
        fileAccessTmp=fileAccessionNb.read()
        for char in fileAccessTmp:
            if ord(char)>128:
                char=''
        fileAccessionNb.close()
        xmlAccessionNb=etree.fromstring(fileAccessTmp) # loading XML file as a tree
        ### Extract Accession number
        names=xmlAccessionNb.findall(".//name")
        numberOfAnnotations+=len(names)
        for name in names:
            if name.text not in accessionNames:
                accessionNames.append(name.text)
        ### Extract Section type
        prefixes=xmlAccessionNb.findall(".//prefix") # find all prefix
        postfixes=xmlAccessionNb.findall(".//postfix") # find all postfix
        sections=xmlAccessionNb.findall(".//section") # find all sections
        subtypes=xmlAccessionNb.findall(".//subType") # find all subtypes
        ### Building a list of pseudo citations which allow after to match with sentences
        preCitPosts=[]
        index=0
        while index<len(names):
            section=''.join(sections[index].itertext())
            if "(" in section:
                section=section.split(" (")[0]
            preCitPosts.append([''.join(prefixes[index].itertext()),''.join(names[index].itertext()),''.join(postfixes[index].itertext()),section,''.join(subtypes[index].itertext())])
            index+=1
        ### Check then the PMCID-sentencized.xml file
        fileSentencized=codecs.open("./articlesOA/"+(str(file).split("-")[0])+"-sentencized.xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        for char in fileSentencizedTmp:
            if ord(char)>128:
                char=''
        fileSentencized.close()
        fileSentencizedTree=etree.fromstring(fileSentencizedTmp) # loading XML file as a tree
        # Remove Tables
        for table in fileSentencizedTree.xpath("//SecTag"):
            if table.attrib["type"]=="TABLE":
                table.getparent().remove(table)
        # Find all sentences for checking..
        sentences=fileSentencizedTree.findall(".//SENT")
        sentencesIndex=0
        while sentencesIndex<len(sentences):
            sentence=''.join(sentences[sentencesIndex].itertext())
            if minlen<len(sentence)<maxlen:
                for preCitPost in preCitPosts:
                    if preCitPost[1] in sentence:
                        indexMatch=sentence.find(preCitPost[1])
                        beginning=indexMatch-20
                        if beginning<0:
                            beginning=0
                        ending=indexMatch+len(preCitPost[1])+20
                        if ending>len(sentence)-1:
                            ending=len(sentence)
                        comparingStrPrefix=sentence[beginning:indexMatch].replace(" ","")
                        comparingStrPostfix=sentence[indexMatch:ending].replace(" ","")
                        PrefixRatio=SequenceMatcher(None,comparingStrPrefix,preCitPost[0].replace(" ",""))
                        PostfixRatio=SequenceMatcher(None,comparingStrPostfix,preCitPost[2].replace(" ",""))
                        citationAnnot=preCitPost[0]+preCitPost[1]+preCitPost[2]
                        if citationAnnot.replace(" ","") in sentence.replace(" ","") or sentence[beginning:ending].replace(" ","") in citationAnnot.replace(" ",""):
                            numberOfExtracted+=1
                            # print (sentence)# function
                            # print ("#####################")
                        elif PrefixRatio.ratio()>threshold and PostfixRatio.ratio()>threshold:
                            numberOfExtracted+=1
                            # print (sentence)# function
                            # print ("*****************************")
                        else:
                            outputmistakefile.write(citationAnnot)
                            outputmistakefile.write("\n")
                            outputmistakefile.write(sentence[beginning:ending])
                            outputmistakefile.write("\n\n")
                            print (citationAnnot)
                            print (sentence[beginning:ending])
                            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sentencesIndex+=1        
        # while sentencesIndex<len(sentences):
        #     for preCitPost in preCitPosts:# for citation
        #         citation=sentences[sentencesIndex]
        #         citationStr=''.join(citation.itertext())
        #         indexMatch=citationStr.find(preCitPost[1])
        #         beginning=indexMatch-19
        #         if beginning<0:
        #             beginning=0
        #         ending=indexMatch+len(preCitPost[1])+20
        #         if ending>len(citationStr)-1:
        #             ending=len(citationStr)
        #         strAnnot=citationStr[beginning:ending]
        #         strRatio=SequenceMatcher(None,strAnnot,preCitPost[0])
        #         scheme=re.escape(preCitPost[0])
        #         if preCitPost[0] in citationStr and minlen<len(citationStr)<maxlen :#or citationStr.find(preCitPost[0])>-1 and minlen<len(citationStr)<maxlen or strRatio.ratio()>0.98 and minlen<len(citationStr)<maxlen:
        #             numberOfExtracted+=1
        #             # remove the "()" part of the string
        #             section=preCitPost[2]
        #             if "(" in section:
        #                 section=section.split(" (")[0]
        #             #check the section of the sentence
        #             for secTag in citation.iterancestors("SecTag"):#citation secTag
        #                 secTag=secTag.get("type")
        #                 if secTag=="FIG":# Assign type of figure or not.
        #                     figure=True
        #                 else:
        #                     figure=False
        #                 break
        #             #check the section of the previous sentence
        #             citationbefore=''
        #             for secTagBefore in sentences[sentencesIndex-1].iterancestors("SecTag"):#citationBefore secTag
        #                 secTagBefore=secTagBefore.get("type")
        #                 break
        #             if secTag==secTagBefore and minlen<len(''.join(sentences[sentencesIndex-1].itertext()))<maxlen:# check if the section of the previous sentence is the same of the citation one
        #                 citationbefore=''.join(sentences[sentencesIndex-1].itertext())
        #             #check the section of the next sentence
        #             citationafter=''
        #             for secTagAfter1 in sentences[sentencesIndex+1].iterancestors("SecTag"):
        #                 secTagAfter1=secTagAfter1.get("type")
        #                 break
        #             if secTag==secTagAfter1 and minlen<len(''.join(sentences[sentencesIndex+1].itertext()))<maxlen:# check if the section of the next sentence is the same of the citation one
        #                 citationafter=citationafter+''.join(sentences[sentencesIndex+1].itertext())
        #                 if sentencesIndex+2<len(sentences):
        #                     for secTagAfter2 in sentences[sentencesIndex+2].iterancestors("SecTag"):
        #                         secTagAfter2=secTagAfter2.get("type")
        #                         break
        #                     if secTag==secTagAfter2 and minlen<len(''.join(sentences[sentencesIndex+2].itertext()))<maxlen:# check if the section of the next next sentence is the same of the citation one
        #                         citationafter=citationafter+''.join(sentences[sentencesIndex+2].itertext())
        #             resultString=''.join(citation.itertext())
        #             resultFile.write(str(file).split("-")[0])# PMCID
        #             resultFile.write("\t")
        #             resultFile.write(preCitPost[1])# AccessionNb
        #             resultFile.write("\t")
        #             resultFile.write(section)# Section
        #             resultFile.write("\t")
        #             resultFile.write(preCitPost[3])# SubType
        #             resultFile.write("\t")
        #             resultFile.write(str(figure))# Figure
        #             resultFile.write("\t")
        #             resultFile.write(citationbefore)# Pre-citation
        #             resultFile.write("\t")
        #             resultFile.write(resultString)# Citation
        #             resultFile.write("\t")
        #             resultFile.write(citationafter)# Post-citation
        #             resultFile.write("\n")
        #     sentencesIndex+=1
resultFile.close()
outputmistakefile.close()
print("There is ",numberOfAnnotations," that are mined by AnnotationAPI.")
print("There is ",numberOfExtracted," citations that have been extracted.")
success=numberOfExtracted/numberOfAnnotations*100
print(success,"% of successfull extracted citations")
print("DONE")