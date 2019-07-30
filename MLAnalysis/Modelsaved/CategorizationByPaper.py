#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 19/07/2019
########################

import pandas
import codecs

################################################    Variables     #################################################
limit=0.95
################################################    Functions     #################################################

###################################################    Main     ###################################################

dataset=pandas.read_csv(
	filepath_or_buffer="Predictions.csv",
	header=0,
	sep="\t")

dicPMCID={}
for index,row in dataset.iterrows():
	# print(row["PMCID"])
	if row["PMCID"] not in dicPMCID:
		dicPMCID.update({row["PMCID"]:{}})
	if row["AccessionNb"] not in dicPMCID[row["PMCID"]]:
		dicPMCID[row["PMCID"]].update({
			row["AccessionNb"]:{
				"crea-count":0,
				"crea-proba":0,
				"back-count":0,
				"back-proba":0,
				"use-count":0,
				"use-proba":0,
				"total":0,
				"sections":[]}})
	if row["Creation"]>limit:
		dicPMCID[row["PMCID"]][row["AccessionNb"]]["crea-count"]+=1
		if row["Creation"]>dicPMCID[row["PMCID"]][row["AccessionNb"]]["crea-proba"]:
			dicPMCID[row["PMCID"]][row["AccessionNb"]]["crea-proba"]=row["Creation"]
	elif row["Use"]>limit and dicPMCID[row["PMCID"]][row["AccessionNb"]]["crea-proba"]<limit:
		dicPMCID[row["PMCID"]][row["AccessionNb"]]["use-count"]+=1
		if row["Use"]>dicPMCID[row["PMCID"]][row["AccessionNb"]]["use-proba"]:
			dicPMCID[row["PMCID"]][row["AccessionNb"]]["use-proba"]=row["Use"]
	elif row["Background"]>limit and dicPMCID[row["PMCID"]][row["AccessionNb"]]["crea-proba"]<limit and dicPMCID[row["PMCID"]][row["AccessionNb"]]["use-proba"]<limit:
		dicPMCID[row["PMCID"]][row["AccessionNb"]]["back-count"]+=1
		if row["Background"]>dicPMCID[row["PMCID"]][row["AccessionNb"]]["back-proba"]:
			dicPMCID[row["PMCID"]][row["AccessionNb"]]["back-proba"]=row["Background"]

	dicPMCID[row["PMCID"]][row["AccessionNb"]]["total"]+=1
	if row["Section"] not in dicPMCID[row["PMCID"]][row["AccessionNb"]]["sections"]:
		dicPMCID[row["PMCID"]][row["AccessionNb"]]["sections"].append(row["Section"])
	

file=codecs.open("Resultbypaper"+str(limit).split(".")[1]+".csv","w",encoding="utf-8")
file.write("PMCID")
file.write("\t")
file.write("AccessionNb")
file.write("\t")
file.write("Crea-Count")
file.write("\t")
file.write("Crea-Proba")
file.write("\t")
file.write("Back-Count")
file.write("\t")
file.write("Back-Proba")
file.write("\t")
file.write("Use-Count")
file.write("\t")
file.write("Use-Proba")
file.write("\t")
file.write("TOTAL")
file.write("\t")
file.write("Sections")
file.write("\n")

for PMCID,dicAccNb in dicPMCID.items():
	# print(PMCID)
	for AccessionNb, dicScores in dicAccNb.items():
		# print("\t",AccessionNb)
		# print("\n",dicScores)
		file.write(PMCID)
		file.write("\t")
		file.write(AccessionNb)
		file.write("\t")
		# dicScores["crea-proba"]=dicScores["crea-proba"]/dicScores["total"]
		# dicScores["back-proba"]=dicScores["back-proba"]/dicScores["total"]
		# dicScores["use-proba"]=dicScores["use-proba"]/dicScores["total"]
		file.write(str(dicScores["crea-count"]))
		file.write("\t")
		file.write(str(dicScores["crea-proba"]))
		file.write("\t")
		file.write(str(dicScores["back-count"]))
		file.write("\t")
		file.write(str(dicScores["back-proba"]))
		file.write("\t")
		file.write(str(dicScores["use-count"]))
		file.write("\t")
		file.write(str(dicScores["use-proba"]))
		file.write("\t")
		file.write(str(dicScores["total"]))
		if dicScores["total"]>1:
			file.write("\t")
			file.write(str(dicScores["sections"]))
		else:
			file.write("\t")
			file.write("")
		file.write("\n")
