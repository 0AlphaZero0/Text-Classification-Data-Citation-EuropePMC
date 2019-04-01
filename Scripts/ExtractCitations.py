#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
from lxml import etree # Allows to manipulate xml file easily
import os # Allows to modify some things on the os
import xml # Allows to manipulate xml files



###################################################    Main     ###################################################
length=(len(os.listdir("./articlesOA"))-1)/2

resultFile=codecs.open("resultCitations.csv","w",encoding="utf-8")
resultFile.write("PMCID")
resultFile.write("\t")
resultFile.write("AccessionNb")
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
        names=xmlAccessionNb.findall(".//name")
        for name in names:
            if name.text not in accessionNames:
                accessionNames.append(name.text)
        fileSentencized=codecs.open("./Sentencized/XML-cured/"+(str(file).split("-")[0])+".xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        fileSentencized.close()
        #os.system('clear')
        #print (str(file).split("-")[0]+".xml")
        fileSentencizedTmp=etree.fromstring(fileSentencizedTmp)
        sentences=fileSentencizedTmp.findall(".//SENT")
        sentencesIndex=0
        while sentencesIndex<len(sentences):
            for accessionNb in accessionNames:
                tmp=sentences[sentencesIndex]
                tmpbefore=''.join(sentences[sentencesIndex-1].itertext())
                if accessionNb in ''.join(tmp.itertext()):
                    tmpafter=''
                    if sentencesIndex+1<len(sentences):
                        tmpafter=tmpafter+''.join(sentences[sentencesIndex+1].itertext())
                        if sentencesIndex+2<len(sentences):
                            tmpafter=tmpafter+''.join(sentences[sentencesIndex+2].itertext())
                    #print ("\n",accessionNb,"|",sentencesIndex)
                    resultString=''.join(tmp.itertext())
                    resultFile.write(str(file).split("-")[0])
                    resultFile.write("\t")
                    resultFile.write(accessionNb)
                    resultFile.write("\t")
                    resultFile.write(tmpbefore)
                    resultFile.write("\t")
                    resultFile.write(resultString)
                    resultFile.write("\t")
                    resultFile.write(tmpafter)
                    resultFile.write("\n")
                    #print (resultFinalString)
                # print ("ERROR")
                # print (file)
                # print (accessionNames)
                # print (accessionNb)
                # print (sentencesIndex)
                # print (sentences[sentencesIndex].get("sid"))
                # print (type(sentences[sentencesIndex].text),"sentences")
            sentencesIndex+=1
resultFile.close()
print("DONE")