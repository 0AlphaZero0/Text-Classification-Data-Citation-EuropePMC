#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 14/05/2019
########################

import codecs

dataset=codecs.open(
    filename="matrice.csv",
    mode="r",
    encoding="utf-8")
pmcid_list=[]
section=[]
for line in dataset.readlines():
    pmcid_list.append(line.split("\t")[0])
    section.append(line.split("\t")[1])
dico={}
for index in range(len(pmcid_list)):
    if pmcid_list[index] in dico:
        dico[pmcid_list[index]][section[index].split('\r')[0]]+=1
    else:
        dico[pmcid_list[index]]={
            # "ArrayExpress":0,
            # "BioModels":0,
            # "BioProject":0,
            # "BioSamples":0,
            # "BioStudies":0,
            # "CATH":0,
            # "ChEBI":0,
            # "ChEMBL":0,
            # "dbGaP":0,
            # "DOI":0,
            # "EBI Metagenomics":0,
            # "EFO":0,
            # "EGA":0,
            # "EMDB":0,
            # "EMPIAR":0,
            # "ENA":0,
            # "Ensembl":0,
            # "EUDRACT":0,
            # "GCA":0,
            # "Gene Ontology (GO)":0,
            # "GEO":0,
            # "HGNC":0,
            # "HPA":0,
            # "IGSR/1000 Genomes":0,
            # "InterPro":0,
            # "MetaboLights":0,
            # "NCT":0,
            # "OMIM":0,
            # "PDBe":0,
            # "Pfam":0,
            # "PRIDE":0,
            # "Reactome":0,
            # "RefSeq":0,
            # "RefSNP":0,
            # "Rfam":0,
            # "RRID":0,
            # "UniProt":0}
            ############################
            # "Use":0,
            # "Unknow":0,
            # # "Compare":0,
            # "Background":0,
            # "Creation":0}
            ############################
            "Abbreviations":0,
            "Abstract":0,
            "Acknowledgments":0,
            "Article":0,
            "Author Contributions":0,
            "Case study":0,
            "Competing Interests":0,
            "Conclusion":0,
            "Discussion":0,
            "Figure":0,
            "Introduction":0,
            "Methods":0,
            "References":0,
            "Results":0,
            "Supplementary material":0,
            "Table":0,
            "Title":0}
        dico[pmcid_list[index]][section[index].split('\r')[0]]+=1
dataset.close()
result=codecs.open(
    filename="matriceresult.csv",
    mode="w",
    encoding="utf-8")
# result.write("PMCID\tAbstract\tAcknowledgments\tArticle\tCase study\tConclusion\tDiscussion\tFigure\tIntroduction\tMethods\tResults\tSupplementary material\tTitle\n")
x=0
result.write("PMCID\t")
for pmcid in dico:
    if x==0:
        for category in dico[pmcid]:
            result.write(category)
            result.write("\t")
        x=1
    result.write("\n")
    result.write(pmcid)
    result.write("\t")
    for category in dico[pmcid]:
        result.write(str(dico[pmcid][category]))
        result.write("\t")
    # result.write(str(dico[pmcid]["Array"]))
    # result.write("\t") 
    ############################################################
    # result.write(str(dico[pmcid]["Use"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Compare"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Background"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Creation"]))
    ###########################################################
    # result.write(str(dico[pmcid]["Abstract"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Acknowledgments"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Article"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Case study"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Conclusion"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Discussion"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Figure"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Introduction"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Methods"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Results"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Supplementary material"]))
    # result.write("\t")
    # result.write(str(dico[pmcid]["Title"]))
    # result.write("\n")