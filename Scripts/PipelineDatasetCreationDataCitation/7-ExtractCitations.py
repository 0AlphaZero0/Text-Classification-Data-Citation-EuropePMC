#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur
########################
import codecs # Allows to load a file containing UTF-8 characters
from lxml import etree # Allows to manipulate xml file easily
import os # Allows to modify some things on the os
import xml # Allows to manipulate xml files

"""
This script take place in a pipeline that extract citation of data in scientific papers, thanks to EuropePMC, RESTful API and Annotation API.

This script will extract citations from XML sentencized files that are located in the articleOA directory. Then it will save those citations in a "pre-dataset" called resultCitations.csv.
It 's organized like this :
PMCID   AccesionNb  Pre-Citation    Citation    PostCitation
_____________________________________________________________
Indeed the context of citation is really important so there is a need to extract those.
In the fist time the script will load a file that end like "-AccessionNb.xml" it contain the Accession numbers of a specific paper (they are find thanks to the lxml etree module : .findall(".//name")).
Then the script will load the corresponding sentencized XML file. If the accession number is in the sentence then the previous sentence and the two after are saved in the file.
"""

###################################################    Main     ###################################################
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
        fileSentencized=codecs.open("./articleOA/"+(str(file).split("-")[0])+"-sentencized.xml","r",encoding="utf-8")
        fileSentencizedTmp=fileSentencized.read()
        fileSentencized.close()
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
            sentencesIndex+=1
resultFile.close()
print("DONE")