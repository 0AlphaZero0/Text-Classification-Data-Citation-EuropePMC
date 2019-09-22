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
import string
import sys # Allows access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import time # Allows to set some point in the execution time and then calculate the execution time

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will extract citations from XML sentencized files that are located in the articlesOA directory. Then it will save those citations in a "pre-dataset" called resultCitations.csv.
It 's organized like this :
-------------------------------------------------------------------------------------------------------
PMCID     AccessionNb     Section     SubType     Figure     Pre-Citation     Citation     PostCitation
_______________________________________________________________________________________________________
Indeed the context of citation is really important so there is a need to extract those.
In the fist time the script will load a file that end like "-AccessionNb.xml" it contain the Accession numbers of a specific paper (they are find thanks to the lxml etree module : .findall(".//name")).
Then the script will load the corresponding sentencized XML file. If the accession number is in the sentence then the previous sentence and the two after are saved in the file.
"""

####################################################    Var    ####################################################
directory=sys.argv[1]
length=(len(os.listdir(directory))-1)/2 # total number of paper to analyze
minlen=25 # character's number minimum in the sentence
maxlen=500 # character's number maximum in the sentence
numberOfExtracted=0 # paper's number that goes through the loop
numberOfAnnotations=0 # annotation's number
threshold=0.85 # treshold for string comparison 
fileNameMistakes="mistakes.csv"
fileNameResult="Result.csv"
#################################################    Function     #################################################

def save(sentences,sentencesIndex,resultFile,preCitPost): # Save citation in file
    '''
    Description : 
        This function save the citation and all metadata in a file
    Args :
        sentences -> list of sentence
        sentencesIndex -> index of sentence containing the exact match
        resultFile -> # path of the resulting file
        preCitPost -> # list with [prefix, exact match, postfix]
    Return :
        No return
    '''
    citation=sentences[sentencesIndex]
    # remove the "()" part of the string
    section=preCitPost[3]
    if "(" in section:
        section=section.split(" (")[0]
    #check the section of the sentence
    secTag=''
    figure=False
    for secTag in sentences[sentencesIndex].iterancestors("SecTag"):#citation secTag
        secTag=secTag.get("type")
        if secTag=="FIG":# Assign type of figure or not.
            figure=True
        else:
            figure=False
        break
    #check the section of the previous sentence
    citationbefore=''
    secTagBefore=''
    for secTagBefore in sentences[sentencesIndex-1].iterancestors("SecTag"):#citationBefore secTag
        secTagBefore=secTagBefore.get("type")
        break
    if secTag==secTagBefore and minlen<len(''.join(sentences[sentencesIndex-1].itertext()))<maxlen:# check if the section of the previous sentence is the same of the citation one
        citationbefore=''.join(sentences[sentencesIndex-1].itertext())
    #check the section of the next sentence
    citationafter=''
    secTagAfter1=''
    secTagAfter2=''
    for secTagAfter1 in sentences[sentencesIndex+1].iterancestors("SecTag"):
        secTagAfter1=secTagAfter1.get("type")
        break
    if secTag==secTagAfter1 and minlen<len(''.join(sentences[sentencesIndex+1].itertext()))<maxlen:# check if the section of the next sentence is the same of the citation one
        citationafter=citationafter+''.join(sentences[sentencesIndex+1].itertext())
        if sentencesIndex+2<len(sentences):
            for secTagAfter2 in sentences[sentencesIndex+2].iterancestors("SecTag"):
                secTagAfter2=secTagAfter2.get("type")
                break
            if secTag==secTagAfter2 and minlen<len(''.join(sentences[sentencesIndex+2].itertext()))<maxlen:# check if the section of the next next sentence is the same of the citation one
                citationafter=citationafter+''.join(sentences[sentencesIndex+2].itertext())
    resultString=''.join(citation.itertext())
    resultFile.write(str(file).split("-")[0])# PMCID
    resultFile.write("\t")
    resultFile.write(preCitPost[1])# AccessionNb
    resultFile.write("\t")
    resultFile.write(section)# Section
    resultFile.write("\t")
    resultFile.write(preCitPost[4])# SubType
    resultFile.write("\t")
    resultFile.write(str(figure))# Figure
    resultFile.write("\t")
    resultFile.write(citationbefore)# Pre-citation
    resultFile.write("\t")
    resultFile.write(resultString)# Citation
    resultFile.write("\t")
    resultFile.write(citationafter)# Post-citation
    resultFile.write("\n")    

###################################################    Main     ###################################################
print("###   5/6 - EXTRACT CITATIONS   ###\n")

start=time.time() # start time
# write firstrow of resulting file
outputmistakefile=codecs.open(fileNameMistakes,"w",encoding="utf-8") # file containing mistakes : exact name find in the sentence but not corresponding to Annotation API citation
resultFile=codecs.open(fileNameResult,"w",encoding="utf-8") # file containing all data citations in all corresponding files
resultFile.write("PMCID\tAccessionNb\tSection\tSubType\tFigure\tPreCitation\tCitation\tPostCitation\n")
# for a PMCID file sentencized :
for file in os.listdir(directory):
    # Check first the PMCID-AccessionNb.xml file
    if file.endswith("-AccessionNb.xml"):
        fileAccessionNb=codecs.open(directory+"/"+str(file),"r",encoding="utf-8")
        fileAccessTmp=fileAccessionNb.read()
        for char in fileAccessTmp:
            if ord(char)>128: # check if it's part of the first 127 unicode characters
                char='' # replace char by '' if not part of 127 first unicode characters
        fileAccessionNb.close()
        xmlAccessionNb=etree.fromstring(fileAccessTmp) # loading XML file as a tree
        ### Extract Accession number
        names=xmlAccessionNb.findall(".//name") # find each accession number provides by Annotation API
        numberOfAnnotations+=len(names) # number of annotation for this file (pmcid) provides by Annotation API
        # create a list of unique accession numbers in the file
        accessionNames=[]
        for name in names:
            if name.text not in accessionNames:
                accessionNames.append(name.text)
        ### Extract Section type
        prefixes=xmlAccessionNb.findall(".//prefix") # find all prefix (~ 10 character before exact name)
        postfixes=xmlAccessionNb.findall(".//postfix") # find all postfix (~ 10 character after exact name)
        sections=xmlAccessionNb.findall(".//section") # find all sections
        subtypes=xmlAccessionNb.findall(".//subType") # find all subtypes
        ### Building a list of pseudo citations (prefix + exact name + postfix)
        preCitPosts=[]
        index=0
        while index<len(names):
            section=''.join(sections[index].itertext())
            if "(" in section:
                section=section.split(" (")[0]
            preCitPosts.append([''.join(prefixes[index].itertext()),''.join(names[index].itertext()),''.join(postfixes[index].itertext()),section,''.join(subtypes[index].itertext())])
            index+=1
        ### Check then the PMCID-sentencized.xml file
        fileSentencized=codecs.open(directory+"/"+(str(file).split("-")[0])+"-sentencized.xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        for char in fileSentencizedTmp:
            if ord(char)>128: # check if it's part of the first 127 unicode characters
                char='' # replace char by '' if not part of 127 first unicode characters
        fileSentencized.close()
        fileSentencizedTree=etree.fromstring(fileSentencizedTmp) # loading XML file as a tree
        # Remove Tables if their are not been removed before
        for table in fileSentencizedTree.xpath("//SecTag"):
            if table.attrib["type"]=="TABLE":
                table.getparent().remove(table)
        # Find all sentences for checking..
        sentences=fileSentencizedTree.findall(".//SENT")
        sentencesIndex=0
        while sentencesIndex<len(sentences)-2: # loop thrgough the sentence list
            sentence=''.join(sentences[sentencesIndex].itertext())
            if minlen<len(sentence)<maxlen: # check if the sentence length is between min and max
                checkedNames=[]
                for preCitPost in preCitPosts: # loop through pseudo citations
                    if preCitPost[1] in sentence and preCitPost[1] not in checkedNames: # check if exact name of pseudo sentence is in sentence from full txt and check if it's not already scan
                        indexMatch=sentence.find(preCitPost[1]) # position of exact match in the string
                        beginning=indexMatch-20
                        if beginning<0:
                            beginning=0
                        ending=indexMatch+len(preCitPost[1])+20
                        if ending>len(sentence)-1:
                            ending=len(sentence)
                        if ending<20:
                            beginning=beginning-(20-ending)
                        comparingStrPrefix=sentence[beginning:indexMatch].replace(" ","") # remove prefix whitespaces
                        comparingStrPostfix=sentence[indexMatch:ending].replace(" ","") # remove postfix whitespaces
                        PrefixRatio=SequenceMatcher(None,comparingStrPrefix,preCitPost[0].replace(" ","")) # compute a score of similarity between prefix from full txt and prefix from pseudo citation
                        PostfixRatio=SequenceMatcher(None,comparingStrPostfix,preCitPost[2].replace(" ","")) # compute a score of similarity between postfix from full txt and postfix from pseudo citation
                        citationAnnot=preCitPost[0]+preCitPost[1]+preCitPost[2] # one string from pseudo citation
                        tmpCitationAnnot=citationAnnot.translate(str.maketrans('', '', string.punctuation))
                        tmpSentence=sentence.translate(str.maketrans('', '', string.punctuation))
                        # check if pseudo citation is exactly in the full txt sentence
                        if tmpCitationAnnot.replace(" ","").replace("�","") in tmpSentence.replace(" ","") or tmpSentence[beginning:ending].replace(" ","") in tmpCitationAnnot.replace(" ","").replace("�",""):
                            numberOfExtracted+=1
                            save(sentences, sentencesIndex,resultFile,preCitPost)
                            checkedNames.append(preCitPost[1])
                        # check if pseudo citation is similar to full txt sentence
                        elif PrefixRatio.ratio()>threshold and PostfixRatio.ratio()>threshold:
                            numberOfExtracted+=1
                            save(sentences, sentencesIndex,resultFile,preCitPost)
                            checkedNames.append(preCitPost[1])
                        else:
                            # write in the mistake file the pseudo citation and coresponding character in the full txt
                            outputmistakefile.write(file.split("-")[0])
                            outputmistakefile.write("\n")
                            outputmistakefile.write(citationAnnot) # pseudo citation
                            outputmistakefile.write("\n")
                            outputmistakefile.write(sentence[beginning:ending]) # full txt sentence (20 characters + exact name +20 charaters)
                            outputmistakefile.write("\n\n")
            sentencesIndex+=1
resultFile.close()
outputmistakefile.close()
end=time.time()
print ("Duration : "+str(int(end-start))+" sec")
print("There is ",numberOfAnnotations," data citaions that are mined by AnnotationAPI.")
print("There is ",numberOfExtracted," data citations that have been extracted.")
success=numberOfExtracted/numberOfAnnotations*100
print(round(success,2),"% of successfull extracted citations")
print("\nExtract citations DONE\n")