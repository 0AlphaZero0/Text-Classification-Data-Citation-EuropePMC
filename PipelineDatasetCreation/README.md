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

### Procedure :

We could specify the directory and the number of paper on which we want to work (1200 papers ~ 2000 citations :arrow_forward: ***1.67 citations/paper***). Then the pipeline will create random PMCID check if the PMCID lead to an *open access* paper, then if it is, it will requests the *annotation API* to return the annotations. If there is no annotations in the paper it will generate another random PMCID. And repeat this until it find one containing data citations. It will then save the full text paper in xML format (PMCXXXXXXX-fulltxt.xml) and save also the annotation file in XML format too (PMCXXXXXXX-AccessionNb.xml). Then it will remove tables from the full text file, indeed tables are problematic for the sentencizer and also we don't need those. Then the sentencizer split the xml file by sentences (there is some warnings but they are not important) and save this in a file(PMCXXXXXXX-sentencized.xml). Then the pipeline will repair some sentences indeed the sentencizer doesn't always split at the good position, so it will be repaired here for some of those. In the end the pipeline will extract sentences and meta data from the sentencized file, it will compare corresponding files thanks to their PMCID (PMCXXXXXXX). Exact match of each accession number in the file are found in the sentencized XML then it will check if the post or pre tag are corresponding thanks to a similarity ratio. Indeed it seems that pre and post tag from annotation API doesn't always corresponds to what is in the full text, so thanks to a treshold parameter we can keep data citation if they are at least 85% similar to the pre-tag + exact-match + post-tag that gives the annotation API.
then it will save all of these in a CSV file. Finally the pipeline remove duplicates, here we call duplicates citations that are in the same sentence, we keep only one of those for example :

*We use **Accession number 1**  and **Accession number 2***

Here we keep only one of those and not the two data citations to avoid the sentence's repetition in the dataset as it could overfit the model.

We can get at the end a dataset called : "Dataset.csv"

It will look like this :

| PMCID      | AccessionNb | Section | SubType    | Figure | Pre-citation                                                                                | Citation                                                                                                                                                                           | Post-citation                                                                                                                                                |
|------------|-------------|---------|------------|--------|------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PMC2928273 | rs2234671   | Methods | RefSNP     | False  | SNP genotypes were called using the GeneMapper software (Applied Biosystems).               | Three SNPs: IL8RA:rs2234671, LTA:rs2229092 and IL4R:rs1805011 were removed because of excessive missing genotypes (>20%).                                                          | All genotyping was completed blinded with regard to toxicity status.                                                                                         |
| PMC2928273 | rs2229092   | Methods | RefSNP     | False  | SNP genotypes were called using the GeneMapper software (Applied Biosystems).               | Three SNPs: IL8RA:rs2234671, LTA:rs2229092 and IL4R:rs1805011 were removed because of excessive missing genotypes (>20%).                                                          | All genotyping was completed blinded with regard to toxicity status.                                                                                         |
| PMC4392464 | PRJNA242298 | Results | BioProject | False  | There were 133 and 50,008 contigs longer than 10,000 and 1,000 bp, respectively (Table 1).  | All assembled sequences were deposited in NCBI’s Transcriptome Shotgun Assembly (TSA) database (http://www.ncbi.nlm.nih.gov/genbank/tsa/) under the accession number PRJNA242298.  | Of the 140,432 contigs, 91,303 (65.0%) had annotation information (Additional file 1: Table S1). For contigs with lengths ≥1,000 bp, 94.7% had BLASTX hits.  |
