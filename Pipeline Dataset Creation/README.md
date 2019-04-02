First the pipeline will execute the ExtractOArticles.py then ExtractPMCID.py then a pmcidlist is given to a sentencizer that will split full txt XML and result a patch.xml file. Then the ExtractSentencizedXML.py and the result of it is given to a tool that I've called RepareXML that correct the structure of the XML file. This tool return XML files corrected but sometimes there still mistakes that has been made by the splitter so the RepareSplitedXML.py will run, followed by ExtractCitations.py and at the end ClearerDataset.py.

Workflow :
1. ExtractOArticles.py
    * ./articlesOA/PMCXXXXXXX-fulltxt.xml
    * ./articlesOA/PMCXXXXXXX-AccessionNb.xml
    * ./articlesOA/PMCYYYYYYY-fulltxt.xml
    * ./articlesOA/PMCYYYYYYY-AccessionNb.xml
    * etc.
2. ExtractPMCID.py
    * ./pmcidList.csv
3. *Sentencizer*
    * ./patch.xml
4. ExtractSentencizedXML.py
    * ./articlesOA/PMCXXXXXXX-sentencized.xml
    * ./articlesOA/PMCYYYYYYY-sentencized.xml
    * etc.
5. *RepareXML*
6. RepareSplitedXML.py
7. ExtractCitations.py
    * ./resultCitations.csv
8. ClearerDataset.py
    * ./articlesOA/datasetX.csv