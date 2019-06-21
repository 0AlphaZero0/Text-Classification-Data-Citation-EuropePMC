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

This pipeline has as a goal to create a csv file that corresponds to a dataset of sentences. Each row should correspond to a data citation in an open access paper from EuropePMC. A data citation here correspond to the sentence containing the accession number. This pipeline should also gives extra features like *PreCitation* which corresponds to the sentence before the data citation. As *PreCitation*, PostCitation corresponds to the sentence after but it also could be two sentences after the citation. Fo those two features it could return empty values. Indeed if the paper's section change between the citation and Precitation or Postcitation it will remove sentences that are not in the same section as the citation. 
The pipeline will also add metadata as the SubType of the citation, the section and if it's in a figure caption or not.

Here is the pipeline diagram :

![](https://github.com/0AlphaZero0/Text-Classification-Data-Citation-EuropePMC/blob/master/PipelineDatasetCreation/ExtractDataCitations.png)

