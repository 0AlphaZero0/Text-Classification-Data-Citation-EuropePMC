<a name="top"></a>
<div class="row">
  <div class="column">
    <img align="left" width="20%" height="20%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/EMBL-EBI-logo.png">
  </div>
  <div class="column">
    <img align="right" width="20%" height="20%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/europepmc.png">
  </div>
</div>
&nbsp;  &nbsp;  &nbsp;  
<h1 align="center">Text Classification on Data Citations in Scientific Papers</h1>
<p align="center">Text Classification on Data Citations in Scientific Papers in Europe Pub Med Central</p>

______________________________________________________________________

## Pipeline for dataset creation

First the pipeline will execute the ExtractOArticles.py then ExtractPMCID.py then a pmcidlist is given to a sentencizer that will split full txt XML and result a patch.xml file. Then the ExtractSentencizedXML.py and the result of it is given to a tool that I've called RepareXML that correct the structure of the XML file. This tool return XML files corrected but sometimes there still mistakes that has been made by the splitter so the RepareSplitedXML.py will run, followed by ExtractCitations.py and at the end ClearerDataset.py.

Workflow :
1. **[ExtractOArticles.py](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Pipeline%20Dataset%20Creation/1-ExtractOArticles.py)**
    * ./articlesOA/PMCXXXXXXX-fulltxt.xml
    * ./articlesOA/PMCXXXXXXX-AccessionNb.xml
    * ./articlesOA/PMCYYYYYYY-fulltxt.xml
    * ./articlesOA/PMCYYYYYYY-AccessionNb.xml
    * etc.
2. **[ExtractPMCID.py](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Pipeline%20Dataset%20Creation/2a-ExtractPMCID.py)**
    * ./pmcidList.csv
3. ***Sentencizer***
    * ./patch.xml
4. **[ExtractSentencizedXML.py](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Pipeline%20Dataset%20Creation/4-ExtractSentencizedXML.py)**
    * ./articlesOA/PMCXXXXXXX-sentencized.xml
    * ./articlesOA/PMCYYYYYYY-sentencized.xml
    * etc.
5. ***RepareXML***
6. **[RepareSplitedXML.py](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Pipeline%20Dataset%20Creation/6-RepareSplitedXML.py)**
7. **[ExtractCitations.py](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Pipeline%20Dataset%20Creation/7a-ExtractCitations.py)**
    * ./resultCitations.csv
8. **[ClearerDataset.py](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Pipeline%20Dataset%20Creation/8-ClearerDataset.py)**
    * ./articlesOA/datasetX.csv
