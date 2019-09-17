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

## Summary & Usefull Link

- *[Description](#Description)*

- *[Organisation](#Organisation)*

- *[Planning](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/projects/1)*

- *[Logbook & Notes](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/README.md)*

- *[Acknowledgment](#Acknowledgment)*

- *[Report](https://github.com/0AlphaZero0/Text-Classification-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Text_classification_EMBL_EBI_EuropePMC.pdf)*

- *[Defense](https://docs.google.com/presentation/d/11LsMi57zz1VA3w30YoLQzFJoa52ZPLCzO-PiH_tW8gA/edit?usp=sharing)*

______________________________________________________________________

<a name="Description"></a>
### Description :

This project is based on [EuropePMC](http://europepmc.org/), this is an on-line database that offers free access to a large and growing collection of biomedical research literature. Actually there is more than 35.3 million of abstract and 5.3 million of full text articles on this database. 

[EuropePMC](http://europepmc.org/) team is part of the [EMBL-EBI](https://www.ebi.ac.uk/), this organisation is itself part of the [EMBL](https://www.embl.org/) organisation. This structure is an international one, composed of 24 country and the goal of this organisation is to develop biology research. It is for this purpose that the EMBL-EBI was created, indeed in the last few years *bioinformatics* had a real impact on biology research, this is due to the fact that the data in this environment has grown in an impressive way, mostly come from genetics. 

The goal of EMBL-EBI is to create open access tools to facilitate the biology research. 
In this perspective a large number of tools have been created in particular:
- [Clustal Omega](http://www.ebi.ac.uk/Tools/msa/clustalo/)
- [GeneWise](http://www.ebi.ac.uk/Tools/psa/genewise)
- [MUSCLE](http://www.ebi.ac.uk/Tools/msa/muscle/)
- [and many others..](https://www.ebi.ac.uk/services/all)
- [etc](https://www.ebi.ac.uk/services/all).

Also there is a lot of data resources that are part of EBI and are currently maintained by EBI teams like :
- [Ensembl](http://www.ensembl.org/)
- [UniProt](http://www.uniprot.org/)
- [PDBe](http://pdbe.org/)
- and of course [Europe PMC](http://europepmc.org/)
- [etc](https://www.ebi.ac.uk/services/all).
 
 In this environnement the Europe PMC team is really close to biomedical scientits so they can answer in the best way biomedical problems for publishing scientific papers. As they deal with millions of papers and try to give the best solution for curators and annotations. 
 
 This last point is one of the reasons why this project is important, indeed analysing millions of papers by just reading them is an exhausting jobs, one of the solution, first, is text-mining, indeed mining for terms like protein name, GO terms, diseases, etc help a lot curators in their jobs. But this just help them and they still need to read a lot to give correct annotations of papers. The solution a recurrent task on millions documents seems pretty famous those last years, Machine Learning & Deep Learning.
 
 So there is a specific field of neuro linguistic programming(NLP), that is Text Classification, this field try, thanks to machine learning and deep learning, to categorize some text thanks to established categories. Indeed today we can transform words in numeric vectors that describe the words and this could be gived to a machine to be learn.
 
The question is "**Is data reuse in scientific documentation**"?

So in this project we will work on data citations, data citations are sentence containing an accession number. An accession number is  an identifier from biological data. Those can be retrieve from Annotation API of Europe PMC in open access papers.

In the first time a dataset has been created thanks to EuropePMC Annotation API, because this API thanks to text mining could return informations about open access and data citations/Accession numbers (and many other things). For this a Pipeline has been created it extract citation and metadata from papers. It looks like :


| PMCID      | AccessionNb | Section | SubType    | Figure | Categories | Pre-citation                                                                                | Citation                                                                                                                                                                           | Post-citation                                                                                                                                                |
|------------|-------------|---------|------------|--------|------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PMC2928273 | rs2234671   | Methods | RefSNP     | False  | Use        | SNP genotypes were called using the GeneMapper software (Applied Biosystems).               | Three SNPs: IL8RA:rs2234671, LTA:rs2229092 and IL4R:rs1805011 were removed because of excessive missing genotypes (>20%).                                                          | All genotyping was completed blinded with regard to toxicity status.                                                                                         |
| PMC2928273 | rs2229092   | Methods | RefSNP     | False  | Use        | SNP genotypes were called using the GeneMapper software (Applied Biosystems).               | Three SNPs: IL8RA:rs2234671, LTA:rs2229092 and IL4R:rs1805011 were removed because of excessive missing genotypes (>20%).                                                          | All genotyping was completed blinded with regard to toxicity status.                                                                                         |
| PMC4392464 | PRJNA242298 | Results | BioProject | False  | Creation   | There were 133 and 50,008 contigs longer than 10,000 and 1,000 bp, respectively (Table 1).  | All assembled sequences were deposited in NCBI’s Transcriptome Shotgun Assembly (TSA) database (http://www.ncbi.nlm.nih.gov/genbank/tsa/) under the accession number PRJNA242298.  | Of the 140,432 contigs, 91,303 (65.0%) had annotation information (Additional file 1: Table S1). For contigs with lengths ≥1,000 bp, 94.7% had BLASTX hits.  |


Some approaches for an automatic classification of those has been studied :
- Tokenization
- Tfidf
- Ngram
- Word Embedding
- Lemmatization
- Stemming
- Models :
  * LSTM
  * CNN
  * A simple neural network (2 dense layers)
  * SVM
  * Complement Naive Bayes
  * Gaussian Naive Bayes
  * Multinomial Naive Bayes
  * Random Forest
  * Logistic Regression
 
______________________________________________________________________


<a name="Organisation"></a>
### Organisation :

This GitHub is ogranized as follows : 
  
- [Datasets](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/tree/master/Datasets)
  * There is two folder in this folder, one that contain [existing datasets](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/tree/master/Datasets/Existing-Dataset) and a second one that contain [dataset created](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/tree/master/Datasets/Dataset-created) for machine learning analysis 

- [Logbook & Notes](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/tree/master/Logbook%20%26%20Notes)
  * In this folder there is all of my notes for the complete internship, there is also a lot of figure that I use in my notes, all of these informations are displayed in the [README.md](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/README.md)

- [ML Analysis](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/tree/master/MLAnalysis)
  * In this folder there is all of the machine learning script and pipeline that I use for an automatic classification of data citations.
  
- [PipelineDatasetCreation](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/tree/master/PipelineDatasetCreation)
  * In this folder there is the pipeline that I use to create a complete dataset of data citations.

There is also a planning that you can find [here](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/projects/1)
______________________________________________________________________

<a name="Acknowledgment"></a>
### Acknowledgment :

For giving me the opportunity to be a part of the great team of EuropePMC, I would like to thank [Johanna McEntyre](https://www.ebi.ac.uk/about/people/johanna-mcentyre) for her advises and her accompaniment throughout this internship.

For his precious advises concerning the annotation part and his accompaniment I would like to thank [Aravind Venkatesan](https://www.ebi.ac.uk/about/people/aravind-venkatesan).

I would like to thank warmly [Xiao Yang](https://www.ebi.ac.uk/about/people/xiao-yang), indeed he is my direct supervisor during this internship, he gives me really good advice. He teaches me a lot and also I would like to thank him for his availability at every moment.

Those three persons were really important for me during this project, indeed they gave me guidelines and their precious helps whenever I needed it. They have been really nice with me and pushing me to gave my best on this project. Therefore, one more time I would like to thank them really much. 

Obviously I would like to thank the EuropePMC team, [Lynne Faulk](https://www.ebi.ac.uk/about/people/lynne-faulk), [Yogmatee Roochun](https://www.ebi.ac.uk/about/people/yogmatee-roochun), [Mariia Levchenko](https://www.ebi.ac.uk/about/people/mariia-levchenko) and [all other members](https://www.ebi.ac.uk/about/people/johanna-mcentyre) for all the goods moments that they gave, without those this internship would not have been the same.

For his precious help I would like to thank [Awais Athar](https://www.ebi.ac.uk/about/people/awais-athar), he gave me his experienced thoughts about this project and gave me really good advice.

In the end I would like to thank my family and especially Claire Pistien for her support and advises during the whole duration of this project.
