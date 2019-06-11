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
<h1 align="center">Loogbook & Notes</h1>
<p align="center">Sentiment Analysis on Data Citations in Scientific Papers in Europe Pub Med Central</p>

______________________________________________________________________

### :closed_book: Meetings

- [05/03 - CambMetrics](#05/03)
- ***[11/03 - Defintions of planinng, needs and requests](#11/03)***
- [12/03 - Questions & path set up](#12/03)
- [19/03 - Dataset conception](#19/03)
- [25/03 - XML Talks & Data creation](#25/03)
- ***[02/04 - Summary of March](#02/04)***
- [16/04 - Pre-Analysis & Discussion with A.Athar](#16/04)
- [24/04 - ML models & Discussion with A.Athar](#24/04)
- [30/04 - ML models & DL models](#30/04)
- [07/05 - LSTM & Embedding](#07/05)
- ***[18/05 - Summary of April](#18/05)***
- [28/05 - Extraction updates](#28/05)

______________________________________________________________________
### :green_book: Notes

- ***[Month 1 - 01/03-31/03](#Month1)***
  * [Week 1 - 01/03-10/03 - Bibliography](#Week1)
  * [Week 2 - 11/03-17/03 - Idea of workflow & Bibliography](#Week2)
  * [Week 3 - 18/03-24/03 - Pre-existing datasets & Approaches](#Week3)
  * [Week 4 - 25/03-31/03 - Dataset extraction pipeline & Bibliography](#Week4)
- ***[Month 2 - 01/04-30/04](#Month2)***
  * [Week 5 - 01/04-07/04 - Dataset extraction pipeline Improvements & Annotations](#Week5)
  * [Week 6 - 08/04-14/04 - Dataset extraction pipeline Improvements & First idea of Categories](#Week6)
  * [Week 7 - 15/04-21/04 - Dataset & Categories & Pre-analysis & First ML models](#Week7)
  * [Week 8 - 22/04-28/04 - ML models](#Week8)
- ***[Month 3 - 29/04-02/06](#Month3)***
  * [Week 9 - 29/04-05/05 - Results & Analysis for ML & DL first models](#Week9)
  * [Week 10 - 06/05-12/05 - Embedding& Cross-validation-score](#Week10)
  * [Week 11 - 13/05-19/05 - TensorBoard & LSTM](#Week11)
  * [Week 12 - 20/05-26/05 - LSTM & Embedding & New data extraction](#Week12)
  * [Week 13 - 27/05-02/06 - Final model building step](#Week13)
  * [Week 14 - 03/06-09/06 - Finishing building models & Optimisation](#Week14)
  * [Week 15 - 10/06-16/06 - ](#Week15)
______________________________________________________________________

### :bookmark_tabs: [Bibliography](#bibliography01)

  *This part contain all link to papers that I've read and my notes about those.*

______________________________________________________________________

### :bookmark_tabs: [Index](#index01)

______________________________________________________________________

### :bookmark: [Attachments](#attachments01)

______________________________________________________________________

## :closed_book: Meetings

<a name="05/03"></a>
### :mega: 05/03 - CambMetrics :

1. **Sentiment Analysis** : *Daniel Encer*

Do it on sicentific data was a big challenge, *:warning: they use someone from outside*, thus they don't really make it, so he can't explain me the all thing.

***Problems encounter*** :
- Many citations in the same sentence (determine which is the target OR source of the citation)
- Hard to have a dataset (there is only few datasets)
- If we build our own dataset it should be random sentence in scientific text to not focus on one field. And to annotate it manually we should use expert in NLP, and of course not only one because the perception of annotation can change according to a personn.

[scite_](https://scite.ai/) : a platform that provide sentiment analysis on scientific papers.


2. **Machine Learning in publishing** : *Colin Batchelor*
- Unsupervised Learning :
  * Clustering
  * Recommendation system
  * Topic modelling
  * Embeddings
- Supervised Learning :
  * Machine translation
  * Facial recognition
  * Speech recognition
  * Speaker identification
- Publishing is a labelling task :
  * Keywords
  * Assigning a journal
  * Reject/Revise/Accept
  * Promote?
  
3. **Document Tiering** : *Peter Corbett*

- Levenstein Edit distance : distance that separate original sentence from a "modified" one : Kitten -> Sitten -> Sittin -> Sitting = distance = 3.
- [spaCy](https://spacy.io/) : NLP toolkit for Python
- Take raw sentences and produce Tokens, Lemmas, Part of speech tags and dependencies
- MultiLayer Perceptron : with 1 hidden layer (12units) relu & adm

4. **Rejected Articles** : *Jeff White*

Know how articles can be rejected. ==> Analyse : Name Author Title

5. **Tools for pre-screening** : *Peter Corbett*

Machine learning task : 
- From title & abstract
- Journal

Prediction : does it pass pre-screening if so does it pass peer review; if not is it eventually accepted by another RSC journal ; how many times is it viewed in the second quarter ; how many times is it cited in its second year?

6. **Is there a publishing gender bias at the RSC?** *Aileen Day*

//

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="11/03"></a>
### :mega: 11/03 - Defintions of planinng, needs and requests :

Meetings :
 - Every Tuesday at 11:00 with Xiao Yang.
 - Every 4 weeks at 11:00 with Xiao Yang & Aravind Venkatesan & Johanna McEntyre.
 
Needs & requests :
 - Logbook needed.
 
 End of July : 
  - Good draft of the internship report.
  - Presentation of the work done

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="12/03"></a>
### :mega: 12/03 - Questions & path set up :

- Answer about questions on the background.
- :warning: Need to set up a problematic!!

Frame the project at high level (can be changed later accordingly, no worries)

What you want to do in the project? What the questions you want to answer in the end?

Examples like:  
What are the sentiments of citations in papers? (must have)  
Do the sentiments vary in different sections?  
Do the sentiments vary with times?  
What are the important features to classify sentiments?  
Which model is better for the classification in terms of accuracy and computation?  

Initial data analysis  
Before the classification, you need to know well about your data  
Look at the data, extract some examples  

Dataset  
Check existing datasets, are they different to our citation task/data?  
Can be use them in our task?  
Build our own datasets if necessary  

Classification  
Are there any challenging problems? ML algorithms etc.  

Results and analysis  

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="19/03"></a>
### :mega: 19/03 - Dataset conception :

- Discussed subjects :
  * Speak about features extraction
  * Dataset conception

- Needs :
  * Mine for citation
  * Need to discuss with Fransesco

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="25/03"></a>
### :mega: 25/03 - XML Talks / Data creation :

- Discussed subjects :
  * XML knowledge (structure, etc..)
  * dataset citations / reference citations

- Needs :
  * Look at lXML & W3sch XML
  * Split sentences using designed tool

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="02/04"></a>
### :mega: 02/04 - Summary of March :

- Subject : 
  * need to focus on data citations to avoid the problem of data OR compare scientific citations with data citations, is there any differences between categories.
  * try to "create" some new categories
 - Explore existing scientific citations sentiment analysis:
   * [\_SCite](https://scite.ai/)
   * [CiTO, the Citation Typing Ontology](https://sparontologies.github.io/cito/current/cito.html)
   * [OpenCitations](http://opencitations.net/)


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="16/04"></a>
### :mega: 16/04 - Pre-Analysis & Discussion with A.Athar :

- Meeting with Xiao Yang :
  * Time and categories for the annotation of the dataset
  * Previous analysis correct 
  * Future work : *Random forest* ; *Naive Bayes* ; *SVM* ; *Logistic Regression*
  * Interesting extracting features to work on :  *Stemming* ; *Tokenization* ; *~~POS tag~~* ; *~~Lexicons~~* ; *N-grams* ; *Word Embedding* ; *Dependency based* ; *Lemma ; *~~Window based Negation~~*

- Meeting with Xiao Yang & Awais Athar : 
  * *ClinicalTrials* are problematic. See with curators from A.Athar team.
  * Doubt about *Creation* category

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="24/04"></a>
### :mega: 24/04 - ML models & Discussion with A.Athar :

- Meeting with Xiao Yang :
  * Need to check MultinomialNB
  * Look at [Glove](https://nlp.stanford.edu/pubs/glove.pdf) for word Embedding
  * Word Embedding with *Multilayer*, *CNN*, *LSTM* classifier

- Meeting with Xiao Yang & Awais Athar : 
  * *ClinicalTrials* should be removed as *Unclassifiable*. For *Background* and *Compare* category curators are not convinced by the utility for data citations.

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="30/04"></a>
### :mega: 30/04 - ML models & DL models :

- *ML models* : Needs to look at stratified 4-fold, because compare category contains only 4 citations.
- *DL models* : Needs to add metrics like precision and recall for a good comparison with scientific works.
  

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="07/05"></a>
### :mega: 07/05 - LSTM & Embedding :

- *Embedding* : Re-work about approaches N-gram/Tokenizer/Lemmatization/Stemming.
  * Indeed I didn't understood that Lemmatization and stemming couldn't be use for a single approach. Stemming or Lemmatization should be used before the tokenizer or N-gram or Embedding. The tokenizer will make each word a token which can be represented by a value or a vector, the N-gram approach will make bag of words and assign a numerical vector to describe it, and the embedding will look at words before and after the current word and represent the word and its context in a numerical vector.
- *LSTM* : It's weird that this Deep learning method gives worst results than embedding sowe need tolook deeper for any problems.
  

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="18/05"></a>
### :mega: 18/05 - Summary of April :

- *Previous analysis* : good but needs to calculate ratio for each paper as it could be really different for each paper (eg. a paper can contain 30 data citation in Results section and another 3 data citations but 1 in Abstract, 1 in Results and 1 in Discussion.)

- *Correction Annotation API* : it forget some citation already match by annotation API, example : if GOxxxx is match one time it would not be match in the paper each time it appears.

- *Expect results soon* : ...

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="28/05"></a>
### :mega: 28/05 - Extraction updates :

- From 800 papers there was ~5000 data citation detected by the Annotation API, then after removing « duplicates » and ClinicalTrials we finally got dataset of ~1500 data citations.
- DOI, DNA & RNA interesting or not??
- Multiple categories in the same sentence

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

______________________________________________________________________

## :green_book: Notes

<a name="Month1"></a>
<a name="Week1"></a>
### :date: Week 1 - 01/03-10/03 - Bibliography :

*Papers & Blogs* :
- [Deep Learning, NLP, Representation](#Deep1)
- [SciLite](#SciLite1)
- [Database citation in full text biomedical papers](#Database1)
- [EuropePMC](#EuropePMC1)
- [Sentiment analysis of scientific citation](#Sentiment1)

<a name="Deep1"></a>
#### :diamond_shape_with_a_dot_inside: [Deep Learning, NLP, Representation](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) :

- :interrobang: ***Problems*** : Why Deep Neural Network work so well?

- :heavy_check_mark: ~~***Solutions***~~ : Resume some questions that has been resolved by deep neural networks (This is a post blog and not a scientific paper)

- :triangular_flag_on_post: ~~***Remaining Problems***~~ :
 
***Softwares & libraries for Text Mining*** :
 - [Textpresso](https://textpresso.yeastgenome.org/textpresso/) --> biology concept
 - [Whatizit](http://www.ebi.ac.uk/webservices/whatizit/info.jsf)
 - [EAGLi](http://eagl.unige.ch/EAGLi/)
 - [Evex](http://evexdb.org/)
 - [Argo](http://argo.nactem.ac.uk/)
 - [Utopia Documents](http://utopiadocs.com/)
 - PubAnnotation
 - Pubator
 - Reflect
 - EXTRACT
 
<a name="SciLite1"></a>
#### :diamond_shape_with_a_dot_inside: [SciLite](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/SciLite%20a%20platform%20for%20displaying%20text-mined%20annotations_A.Venkatesan_et_al.pdf) :

***SciLite: a platform for displaying text-mined annotations as a means to link research articles with biological data.*** A.Venkatesan et al.

- :interrobang: ***Problems*** : Need to link literature and underlying data.

- :heavy_check_mark: ***Solutions*** : Creation of SciLite, a platform that integrates text-minned annotations from different sourcesand overlays those outputs on research articles. 

- :triangular_flag_on_post: ***Remaining Problems*** : Sometimes it's not enough accurate, it affect the trust in the tool.

Platform that allow text mining annotation from any provider to be highlighted on scientific papers.
Make deeper links between literature and biological data sources thanks to text mining.


 - ELIXIR platform/service
 - Work on full articles (~900 000 articles)
 - Annotation stored on MangoDB
 - Resource Description(RDF) : graph model that describe in a formal way WEB ressources.
 
 Annotation types :
  - Named entities (genes, protein names, organisms, diseases, GO terms, chemicals, accession numbers)
  - Biological events (phosphorylation)
  - Relationships (Target-Disease)
  - Text phrases (Gene RIF : "*Gene Referencing to a Function*"; Molecular Interaction)

Output :
![](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/SciLite.PNG)

Workflow : 
 1) Retrieving all anotations for a specific PMCID from the anotation databaseusing API request
 2) Sorting response to their position in text
 3) Display information

![](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/EPMC-SciLite.png)

<a name="Database1"></a>
#### :diamond_shape_with_a_dot_inside: [Database citation in full text biomedical papers](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Database%20Citation%20in%20Full%20Text%20Biomedical%20Articles_S.Kafkas_et_al.pdf) :

***Database Citation in Full Text Biomedical Articles*** S.Kafkas et al.

- :interrobang: ***Problems*** : Are supplementary files necessary bring more informations to biomolecular databank?

- :heavy_check_mark: ***Solutions*** : Use text mining in scientific papers significantly enrich publishers’ annotations and contribute to literature–database cross links (integration in EuropePMC) / WhatizitANA pipeline

- :triangular_flag_on_post: ***Remaining Problems*** : Need solution for ENA because they doesn't use RefSeq. 

*EPMC* :heavy_plus_sign: *NLM* (National Library of Medicine) :arrow_forward: .XML thanks to OCR

Identification oof citation thanks to *WhatizitANA* pipeline

***Problems*** : 
 - Footnotes
 - *ENA* does not include RefSeq
 - Sometimes the pipeline partially identifies accession numbers
 - other errors from wrong assignation of ID like *GenBank* instead of *UniProt*

<a name="EuropePMC1"></a>
#### :diamond_shape_with_a_dot_inside: [EuropePMC](http://europepmc.org/) (EPMC/Europe PubMed Central) [PDF](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Database%20citation%20in%20supplementary%20data%20linked%20to_S.Kafkas_et_al.pdf):

***Database citation in supplementary data linked to Europe PubMed Central full text biomedical articles*** S.Kafkas et al.

- :interrobang: ***Problems*** : Linking scientific literature to databases(DB). How (text)txt mining bring more information to citations.

- :heavy_check_mark: ***Solutions*** : Accession number identification by Whatizit through the Whatizit ANA pipeline.

- :triangular_flag_on_post: ***Remaining Problems*** : More accurate to do it manualy BUT take a lot of time and there is also really good result for automatic way ==> it need some improvements.

PMID = ID for non full text papers (Abstract only) / PMCID = ID for full text papers
 
Pipeline EPMC's : text mining to extract accession references
perform with [STATA](https://www.stata.com/) (statistical tool)
![](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/EuropePMCschema1.PNG)

Features : Accession ID; Deposition ID; Deposition Year; First Public Year; PMID publication year; citation year; citation

<a name="Sentiment1"></a>
#### :diamond_shape_with_a_dot_inside: [Sentiment analysis of scientific citation](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Sentiment%20analysis%20of%20scientific%20citations_A.Athar_et_al.pdf) :

***Sentiment analysis of scientific citation*** A.Athar

- :interrobang: ***Problems*** : 
  * Many citations in the same sentence
  * Sentiment in citation are often hidden
  * Citations are often neutral (description of a method) ==> Subjective / Objective
  * Much variation between scientific text and other genres on lexical terms (lexicon, eg : "surprising movie" / "surprising results")
  * Sentiment could be in citation sentence but also arround (like in the next 3 ones)

- :heavy_check_mark: ***Solutions*** : 
  * To sentiment hidden problem + scientific terms : supervised learning on scientific sentences.
  * Neutral citations are exclude
  * Sentiment could be in citation sentence but also arround (like in the next 3 ones) : He used the context to improve his results

- :triangular_flag_on_post: ***Remaining Problems*** : Anaphore aren't resolved(Lee et al.,2011), lexical chain (Barzilay et Elhadad,1997)should help, entity coherence(Barzilay et Lapata,2008)
 
--> Focused on identify the importance of a citation in a paper ==> 9% are really important

*Supervised learning VS. Unsupervised learning* :

Supervised learning is longer and harder because if there is no annotated dataset already existing, it should be create.
BUT it's a better and specific way to do.

*Categories and Granularity of Sentiment* :
 - positive VS. negative
 - positive VS. neutral VS. negative
 - subjective VS. objective
 - neutral VS. low VS. medium VS. high

==> Granulartity is usefull to set a medium level of criticism

Things that are used for sentiment analysis :
 - adjective/adverbs
 - word meaning
 - level of sentence
 - complete document
 
**Workflow** :
![alt text](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/A.Atharschema1.PNG)
***Sentiment Analysis***

Sentiment with multiple sentiments in the same sentence ==> he decide to take the last one (eg : ...but...not...but[...])

To say that a paper or sentence or opinion is positive we can :
 - directly (adjectives/adverbs)
 - comparison : "paper (of the citation) is better than ..."
 - comparison : "paper (of the citation) improves ..."

Citations with no snetiment are considered as OBJECTIVES.

:warning: Content is really important ==> 328% more negative sentences + 100% more positive sentences

***Classification*** = **SVM with features : "1-3 grams + Dependency features + Window Based Negation"**

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week2"></a>
### :date: Week 2 - 11/03-17/03 - Idea of workflow & Bibliography :

- *Papers & Blogs* :
  * [Measuring the Evolution of a Scientific Field through Citation Frames](#Measuring2)

- *Done* :
  * Creation of the [GitHub](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC)

- *Tricks* :
  * [Scikit-Learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) work with text data
  * Maybe use letters to editor????

- *:warning: Raised problems* :
  * Retracted paper should be negatives

**Idea of workflow** :

So questions that I want to respond are :
- *Is there an impact of the time or publication journals on the opinion of an article or a subject or an author, etc.?*
- *What are the important features , which model is better for the classification in terms of accuracy and computation?*
- *Problems known* :
  * As I know there is not much comparative studies to compare models and/or features
  * There is in a lack of datasets and actually 


![alt text](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/WorkflowIdea.png)

*The idea is to take a paper and extract all citations of this one, to apply sentiment analysis on, then using the measured polarity we can, thanks to meta data, see if the opinion of this article varies over time or over publication venues. We can also move to the higher level of abstraction and not use only one article but some "linked articles". Like all articles for the same author or articles speaking of cancer Vs tuberculosis, etc. So we could see the "general" opinion to articles related to an author or a subject.*

<a name="Teufeljson"></a>
In fact there is a lack of data to this problem only *"Measuring the Evolution of a Scientific Field through Citation Frames D.Jurgens et al."* proposed a really good dataset [Teufeljson](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Datasets/teufel-json.tar.gz) thanks to ACL Anthology Network Corpus. But there is some problems here, this dataset is quiet heavy due to information duplication. Also, this dataset isn't annotated with polarity it's only with 6 classes that I described after.

<a name="CitationSentimentCorpus"></a>
There is also the dataset : [Citation sentiment corpus](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Datasets/Citation_sentiment_corpus.zip), this one look really good, indeed it use objective/positive/negative classes, but one problem is that we don't work with positive or negative labels.


***One solution could be to create my own dataset***


<a name="Measuring2"></a>
#### :diamond_shape_with_a_dot_inside: [Measuring the Evolution of a Scientific Field through Citation Frames](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Measuring%20the%20Evolution%20of%20a%20Scientific%20Field%20through%20Citation%20Frames_D.Jurgens_et_al.pdf) :

***Measuring the Evolution of a Scientific Field through Citation Frames*** D.Jurgens et al.

- :interrobang: ***Problems*** : 
  * We know relatively little about how citation frames develop over time within a field and what impact they have on scientific uptake
  * There is a lack of dataset showing how citations function at the field scale

- :heavy_check_mark: ***Solutions*** : 
  * They perform the first field-scale study of citation framing by first de- veloping a state-of-the-art method for automatically classifying citation function and then applying this method to an entire field’s literature to quantify the effects and evolution of framing
  * Creation of one of the largest annotated corpora of citations and use it to train a high-accuracy method for automat- ically labeling a corpus
  
- :triangular_flag_on_post: ***Remaining Problems*** : 
  * Seems like they didn't use sentences before the citation and this can cause a wrong classification
  (eg. *"BilderNetle is our new data set of German noun-to- ImageNet synset mappings. ImageNet is a large- scale and widely used image database, built on top of WordNet, which maps words into groups of im- ages, called synsets (Deng et al., 2009)."*)

1. They introduce a new large-scale representative corpus of citation function and state-of-the-art methodology for classifying citations by function.

2. They demonstrate that citations reflect the discourse structure of a paper but that this structure is significantly influenced by publication venue.

3. They show that differences in a paper’s citation framing have a significant and meaningful impact on future scientific uptake as measured through future citations.

4. By examining changes in the usage of citation functions, they show that the scholarly NLP community has evolved in how its authors frame their work, reflecting the maturation and growth of the field as a rapid discovery science.

They use 6 categories to classify citations :
 - Background (*P provides relevant information for this domain*)
 - Motivation (*P illustrates need for data, goals, methods, etc.*)
 - Uses (*Uses data, methods, etc. from P*)
 - Extension (*Extends P's data, methods, etc.*)
 - Comparison OR Contrasts (*Express similarity/differences to P*)
 - Future (*P is a potential avenue for future work*)

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week3"></a>
### :date: Week 3 - 18/03-24/03 - Pre-existing datasets & Approaches:

- *Papers & Blogs* :
  * [New Features for Sentiment Analysis: Do Sentences Matter?](#NewFeatures3)
  * [Sentiment Symposium Tutorial](http://sentiment.christopherpotts.net/index.html)
  * [AUTOLEX: An Automatic Lexicon Builder for Minority Languages Using an Open Corpus](#AUTOLEX3)

There is some datasets that already exists for sentiment analysis like :
 * [***Sentiment140***](http://help.sentiment140.com/for-students/), this dataset is really popular because it contains 160,000 tweets collected via Twitter API (unfortunately emoticons are pre-removed, there is actually some research about sentiment analysis using emoticons and they provide a lot of good results), unfortunately this dataset can't be used here, because we're focusing on scientific paper and not on tweets.
 * [***Stanford Sentiment Treebank***](https://nlp.stanford.edu/sentiment/code.html), this dataset like the precedent one is not focusing on scientific papers (entertainment review website) so we're not using it.
 * [***Multidomain sentiment analysis dataset***](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/), this dataset is a multi domain-one (Amazon reviews), but as I can read in many papers, it's actually quite hard to obtain good results with multi-domain, so we're focusing only in scientific papers.
 * [***Paper Reviews Data Set***](https://archive.ics.uci.edu/ml/datasets/Paper+Reviews), this dataset could be great because it contains scientific papers reviews, but it's only reviews and only focusing on computing and informatic, the goal of this project is to apply sentiment analysis on biomedical scientific papers, so this one is not really adapted for the task.
 * [***Awais Athar - Citation Sentiment Corpus***](http://cl.awaisathar.com/citation-sentiment-corpus/), this dataset that I've find in the thesis of Awais Athar, is quite good but as said [earlier](#CitationSentimentCorpus), it's maybe not the perfect one to use here.

One of the solution is to ***create our own dataset***, in fact EPMC provides a lot of abstract and full-text biomedical papers and it could be really easy to extract them, thanks to requests with [RESTful API](http://europepmc.org/RestfulWebService). And another really good point of this method is that we obtain these files in XML format and it's acutally really helpfull to investigate for citations.

So to create our own dataset it's important to know which features are necessary for sentiment analysis.
It's important to separate subjective sentences from objectives one to avoid problems of data size and to avoid time computation for nothing. One of the best approach for now is the *dependency* one.

- Approaches for sentiment analysis :
  1. ***Stemming*** :
In linguistic morphology and information retrieval, stemming is the process of reducing inflected words to their word stem, base or root form generally a written word form.
The original goal of this approach in sentiment analysis is to remove the noise that produce a lot of word's different shapes.
*Problems* : sometimes algorithm just take away the sentiment of the word eg. : ***captivation/captive became both "captiv"*** and also it could be costly in resources and performance accuracy.
So it may not be a really good approach for sentiment analysis.

  2. **Tokenization** :
Tokenization is the process of demarcating and possibly classifying sections of a string of input characters. The resulting tokens are then passed on to some other form of processing. The process can be considered a sub-task of parsing input ( ***"The sky is grey" become "the","sky","is","grey"*** ).
This solution could be great if there is not as much data but if there is a lot of data it could be more complicated to see a really good improvement thanks to this approach (1500 samples seems to be the limitation between "necessary" and "unecessary").

  3. **Part of speech tagging (POStag)** :
In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST), also called grammatical tagging or word-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context—i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. A simplified form of this is commonly taught to school-age children, in the identification of words as nouns, verbs, adjectives, adverbs, etc. (see [POStag scheme](#POStag01))

  4. **Lexicons** :
A lexicon, word-hoard, wordbook, or word-stock is the vocabulary of a person, language, or branch of knowledge (such as nautical or medical). In linguistics, a lexicon is a language's inventory of lexemes. There is two major lexicons that are [SentiWordNet](https://github.com/aesuli/sentiwordnet) and [WordNet](https://wordnet.princeton.edu/). The problem with these two is that these two lexicons are made for general sentiment analysis that is proved to not be efficient when we are interest in a specific field like science.

  5. **N-grams** :
In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus. When the items are words, n-grams may also be called shingles. The three-gram seem to be a really good one (see [A.Athar thesis](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Sentiment%20analysis%20of%20scientific%20citations_A.Athar.pdf)) eg. ***San Francisco (is a 2-gram), The Three Musketeers (is a 3-gram), She stood up slowly (is a 4-gram)*** .

  6. **Word Embedding (word vectors)** :
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.
There is actually a problem with this approach (Agree/disagree are two words that are considered like close ones) but I've read that there is many solutions to that problem. [***Word2Vec***](https://arxiv.org/abs/1301.3781) & [***Glove paper***](https://nlp.stanford.edu/pubs/glove.pdf) & [***Glove website***](https://www.youtube.com/redirect?event=video_description&v=5PL0TmQhItY&redir_token=BRkdmKDmeBCbACo0ewVKUaj5e6t8MTU1MzA4MDAzMkAxNTUyOTkzNjMy&q=https%3A%2F%2Fnlp.stanford.edu%2Fprojects%2Fglove%2F)

  7. **Dependency based** :
Dependency grammar (DG) is a class of modern grammatical theories that are all based on the dependency relation (as opposed to the relation of phrase structure) and that can be traced back primarily to the work of Lucien Tesnière. Dependency is the notion that linguistic units, e.g. words, are connected to each other by directed links. The (finite) verb is taken to be the structural center of clause structure. All other syntactic units (words) are either directly or indirectly connected to the verb in terms of the directed links, which are called dependencies. DGs are distinct from phrase structure grammars, since DGs lack phrasal nodes, although they acknowledge phrases. Structure is determined by the relation between a word (a head) and its dependents. Dependency structures are flatter than phrase structures in part because they lack a finite verb phrase constituent, and they are thus well suited for the analysis of languages with free word order, such as Czech, Slovak, and Warlpiri.
See these [trees](#dependencytrees3)
([Paper](https://www.semanticscholar.org/paper/Concept-Level-Sentiment-Analysis-with-Semantic-A-Agarwal-Poria/6698c5848bb91c8f702994a1ea43b73df8b0aea9?navId=paper-header))

  8. **Lemma** :
Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
One solution here is to lemmatized n-grams and use this as a feature. See [4.2.7 of A.Athar Thesis](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Sentiment%20analysis%20of%20scientific%20citations_A.Athar.pdf)
In computational linguistics, lemmatisation is the algorithmic process of determining the lemma of a word based on its intended meaning. Unlike stemming, lemmatisation depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document. As a result, developing efficient lemmatisation algorithms is an open area of research.

  9. **Window based Negation/ Window based scope** :
  There has been much work in handling negation and its scope in the context of sentiment classification (Polanyi and Zaenen, 2006; Moilanen and Pulman, 2007). Detection of both negation and its scope are non-trivial tasks on their own. Das and Chen (2001) use a window-based approach, where they orthographically modify all words within a fixed window which follow a negation word. Councill et al. (2010) use features from a dependency parser in a CRF framework to detect the scope of negation. More recently, Abu Jbara and Radev (2012) propose a similar framework, but with lexical, structural, and syntactic features while solving a shared task for resolving the scope and focus of negation.
  
<a name="NewFeatures3"></a>
#### :diamond_shape_with_a_dot_inside: [New Features for Sentiment Analysis: Do Sentences Matter?](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/New%20Features%20for%20Sentiment%20Analysis_G.Gezici_et_al.pdf) :

***New Features for Sentiment Analysis: Do Sentences Matter?*** G.Gezici et al.

- :interrobang: ***Problems*** : 
  * Collecting a large training data is often a problem, because of the time a big part of the dataset is just subjective sentences.

- :heavy_check_mark: ***Solutions*** : 
  * Create 19 features to obtain only subjectives sentences related to the review. 
  
- :triangular_flag_on_post: ***Remaining Problems*** : 
  * Not all set of features work well.

Existing solutions instead of the one here :
- Finding subjective sentences
- Exploit the strucure in sentences
- Polarity buit determine which one after
- First review line

19 features split in 4 categories :
- Basic features
- Features based on subjective sentence occurrence statistics
- Delta-tf-idf weighting of word polarities
- Sentence-level features

They also use Punctuation but useless in scientific paper.

<a name="AUTOLEX3"></a>
#### :diamond_shape_with_a_dot_inside: [AUTOLEX: An Automatic Lexicon Builder for Minority Languages Using an Open Corpus](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/AUTOLEX%20An%20Automatic%20Lexicon%20Builder%20for%20Minority%20Languages_E.L.C.Buhay_et_al%20.pdf) :

***AUTOLEX: An Automatic Lexicon Builder for Minority Languages Using an Open Corpus*** E.L.C.Buhay_et_al.

- :interrobang: ***Problems*** : 
  * It's actually hard to obtain a lexicon in Philippine natural language and there it's indeed tedious to create a lexicon manually.

- :heavy_check_mark: ***Solutions*** : 
  * They create AUTOLEX based on other lexicon builder to other language, like arabic.
  
- :triangular_flag_on_post: ***Remaining Problems*** : 
  * There is a remaining problem in all languages, it's the  domain specific problem. When a speech spoke about a specific domain, a same sentence could result in a different meaning for another domain.
  
This article show how to build a lexicon automaticaly so it could be a solution, to do the sentiment analysis.

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week4"></a>
### :date: Week 4 - 25/03-31/03 - Dataset extraction pipeline & Bibliography:

During this week I've start to create a pipeline that can extract data citations from EPMC. It should be completed at the beginning of the next week. ([Week 5](#Week5))

- *Papers & Blogs* :
  *  Data Citation Synthesis Group: Joint Declaration of Data Citation Principles. Martone M. (ed.) San Diego CA: FORCE11; 2014 https://doi.org/10.25490/a97f-egyk
  *  [Achieving human and machine accessibility of cited data in scholarly publications](#Achieving4)

<a name="Achieving4"></a>
#### :diamond_shape_with_a_dot_inside: [Achieving human and machine accessibility of cited data in scholarly publications](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Achieving%20human%20and%20machine%20accessibility%20of%20cited%20data%20in%20scholarly%20publications_J.Starr_et_al%20.pdf) :

***Achieving human and machine accessibility of cited data in scholarly publications*** J.Starr et al.

- :interrobang: ***Problems*** : 
  * This paper show the problem that there is a need of a standard to cite data

- :heavy_check_mark: ***Solutions*** : 
  * So it make a summary on what exist and the pros and cons
  
- :triangular_flag_on_post: ***Remaining Problems*** : 
  * There is still data that ot follow standard of today but most of these are realtively old.

Moving to a cross-discipline standard for acknowledging the data allows researchers to justify continued funding for their data collection efforts (Uhlir, 2012; CODATA-ICSTI Task Group , 2013)

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Month2"></a>
<a name="Week5"></a>
### :date: Week 5 - 01/04-07/04 - Dataset extraction pipeline Improvements & Annotations :

I've try to improve a kind of **pipeline to create dataset for the future analysis**, for this First I extract papers that are open access and contains Accession Numbers so there is a full-txt and an Accession-Numbers file in XML format from the annotation API and RESTful API. Then I extract a PMCID list that it used to "sentencized" full-txt XML files. Then I use a little script to repare result files, indeed the splitter sometimes made mistakes, And it's kind of easy to fix some of these. Then I extract citation from those repared files and it result in a file with all sentences containing all citations. But we're focusing on citations that are currently mine by the annotation API.

To create a good dataset we decide to take the **section** and the **subtype** of the citation eg. section= *Results* or *Methods* and SubType= *ENA* or *PDBe*.

We also decide to **remove citations** that have less than 25 characters and more than 500 characters, indeed those with less than 25 are most of the time title like *INTRODUCTION* or *Suplementary-material*.
Thanks to a little analysis we fix those two limits indeed most of the length of citations and context sentences are between 25 and 800. But those which are mined start from 1 to 30000 characters.

And I've also notice that citations are mostly at the end of a paragraph unlike the beginning of it.

It could be great to add a feature **Figure** that can be set to *True* or *False* or reaplce the section feature (Abstract, Methods etc. by *Figure* when the citation take place in a caption's figure.

At the end I've start to annotate ~1000 citations with categories : **Background, Use, ClinicalTrials, Creation **

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week6"></a>
### :date: Week 6 - 08/04-14/04 - Dataset extraction pipeline Improvements & First idea of Categories :

I had a features called **Figure** which can be *True* or *False* because it seems like if a data citation is in a caption's figure, so there is high chance to be use in the paper.

I've also implemented **non-ascii character suppression**, because it looks like sometimes it just corrupt a complete sentence even for an human reading.

I thought to **other categories** because sometimes the limit is really blurry :

 - Unclassifiable/other : it take references and sentences that doesn't make any that look like false positive.
 - Study : here authors have study the current data
 - Match : here they don't really use the data but match them like with a BLASTX or BLASTN etc.
 - Design/Inspire : I've note that a lot of authors told that they design thanks to a dataset something, it's a background thing but maybe it could be split between thses two.

:warning:I've note something that can be helpfull : sometimes citations looks like : *AccessionNb **to** AccessionNb+n*, so I think it mean that authors use AccessionNb, AccessionNb+1, AccessionNb+2, AccessionNb+3,... AccessionNb+n where n is the last of the series., so in one citation there is sometimes more AccessionNb than what is written.

Fix the extract citation script that were bugged for the section assignation. ~800 citations extracted / ~1200 txt-mined by [AnnotationAPI](http://europepmc.org/AnnotationsApi) / ~65% / ~2000 exact match (citations that are not extracted are citations from tables, etc..)

There is a big difference between citations from Annotation match and my match of accession number from AnnotationAPI because it need to go over different filters and validate each one. 


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week7"></a>
### :date: Week 7 - 15/04-21/04 - Dataset & Categories & Pre-Analysis & First ML models :

I've made a previous data analysis that can be seen [below](#previousanalysis07). 

Also in the end I've implemented first machine learning test, [here](#firstresult07) is first results. It as been made with the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) of [Scikit-learn](https://scikit-learn.org/stable/index.html).

:warning: **Surprisingly giving the vectorizer the complete set of features gives lower result, than just giving the sentence of the citation. And also break Naive bayes approach as we can see with *MultinomialNB*.**

<a name="previousanalysis07"></a>
#### :bar_chart: Analysis :

The final dataset or the pipeline after annotation will look like :

| PMCID      | AccessionNb | Section | SubType    | Figure | Categories | Pre-citation                                                                                | Citation                                                                                                                                                                           | Post-citation                                                                                                                                                |
|------------|-------------|---------|------------|--------|------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PMC2928273 | rs2234671   | Methods | RefSNP     | False  | Use        | SNP genotypes were called using the GeneMapper software (Applied Biosystems).               | Three SNPs: IL8RA:rs2234671, LTA:rs2229092 and IL4R:rs1805011 were removed because of excessive missing genotypes (>20%).                                                          | All genotyping was completed blinded with regard to toxicity status.                                                                                         |
| PMC2928273 | rs2229092   | Methods | RefSNP     | False  | Use        | SNP genotypes were called using the GeneMapper software (Applied Biosystems).               | Three SNPs: IL8RA:rs2234671, LTA:rs2229092 and IL4R:rs1805011 were removed because of excessive missing genotypes (>20%).                                                          | All genotyping was completed blinded with regard to toxicity status.                                                                                         |
| PMC4392464 | PRJNA242298 | Results | BioProject | False  | Creation   | There were 133 and 50,008 contigs longer than 10,000 and 1,000 bp, respectively (Table 1).  | All assembled sequences were deposited in NCBI’s Transcriptome Shotgun Assembly (TSA) database (http://www.ncbi.nlm.nih.gov/genbank/tsa/) under the accession number PRJNA242298.  | Of the 140,432 contigs, 91,303 (65.0%) had annotation information (Additional file 1: Table S1). For contigs with lengths ≥1,000 bp, 94.7% had BLASTX hits.  |

As we can see with this first analysis we can note that the category *use* is most represented with almost 63% of all citations, then there is 11.92% of *creation* of data, 11.63% of *background* and also 9.07% of *ClinicalTrials*.
We decide to create a specific category for *clinicalTrials* indeed it's hard to say if the clinicalTrials is use or create, so we decide to create this.

We can also see that there is most of citations in the section *result* and *methods*. Those two categories are followed by the *article* section, it's a category that seems pretty weird and I think it seems most of the time like a *background/use/creation* section. So this section could be a noisy for the future algorithm. Then the other categories could explain their score by the fact that they are rare for some of these like *conclusion* or *case study*.



- Categories repartition :

| Categories     | Count of Categories | Percentage |
|----------------|---------------------|------------|
| Background     | 159                 | 11.63%     |
| ClinicalTrials | 124                 | 9.07%      |
| Compare        | 4                   | 0.29%      |
| Creation       | 163                 | 11.92%     |
| Unclassifiable | 56                  | 4.10%      |
| Use            | 861                 | 62.98%     |
| TOTAL          | 1367                | 100.00%    |

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/CategoriesRepartitionCircle.PNG">
</p>

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/CategoriesRepartitionGraphic.PNG">
</p>

- Sections repartitions :

| Section                | Count   of Section | Percentage |
|------------------------|--------------------|------------|
| Abstract               | 58                 | 4%         |
| Acknowledgments        | 4                  | 0%         |
| Article                | 170                | 12%        |
| Case study             | 8                  | 1%         |
| Conclusion             | 8                  | 1%         |
| Discussion             | 92                 | 7%         |
| Figure                 | 91                 | 7%         |
| Introduction           | 78                 | 6%         |
| Methods                | 390                | 29%        |
| References             | 12                 | 1%         |
| Results                | 435                | 32%        |
| Supplementary material | 19                 | 1%         |
| Title                  | 2                  | 0%         |
| TOTAL                  | 1367               | 100%       |

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/SectionsRepartitionCircle.PNG">
</p>

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/SectionsRepartitionGraphic.PNG">
</p>

- FIGURE features repartitions :

| Figure | Count   of Figure | Percentage |
|--------|-------------------|------------|
| FALSE  | 1188              | 87%        |
| TRUE   | 179               | 13%        |
| TOTAL  | 1367              | 100%       |

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/FigureRepartitionGraphic.PNG">
</p>

- SubType repartitions :

| SubType            | Count   of SubType | Percentage |
|--------------------|--------------------|------------|
| ArrayExpress       | 5                  | 0%         |
| BioProject         | 23                 | 2%         |
| dbGaP              | 5                  | 0%         |
| DOI                | 25                 | 2%         |
| EMDB               | 1                  | 0%         |
| ENA                | 392                | 29%        |
| Ensembl            | 3                  | 0%         |
| EUDRACT            | 2                  | 0%         |
| GCA                | 6                  | 0%         |
| Gene Ontology (GO) | 46                 | 3%         |
| GEO                | 31                 | 2%         |
| HGNC               | 2                  | 0%         |
| HPA                | 2                  | 0%         |
| IGSR/1000 Genomes  | 7                  | 1%         |
| InterPro           | 4                  | 0%         |
| NCT                | 123                | 9%         |
| OMIM               | 51                 | 4%         |
| PDBe               | 376                | 28%        |
| Pfam               | 18                 | 1%         |
| PRIDE              | 2                  | 0%         |
| RefSeq             | 26                 | 2%         |
| RefSNP             | 164                | 12%        |
| RRID               | 7                  | 1%         |
| UniProt            | 46                 | 3%         |
| TOTAL              | 1367               | 100%       |

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/SubTypeRepartitionGraphic.PNG">
</p>

- Repartitions by categories :

| Categories     | Abstract | Acknowledgments | Article | Case study | Conclusion | Discussion | Figure | Introduction | Methods | References | Results | Supplementary material | Title |
|----------------|----------|-----------------|---------|------------|------------|------------|--------|--------------|---------|------------|---------|------------------------|-------|
| Background     | 1        | 0               | 44      | 0          | 2          | 34         | 1      | 29           | 17      | 0          | 30      | 1                      | 0     |
| ClinicalTrials | 31       | 1               | 39      | 0          | 1          | 7          | 0      | 13           | 30      | 0          | 2       | 0                      | 0     |
| Compare        | 0        | 0               | 0       | 0          | 0          | 0          | 0      | 0            | 0       | 0          | 4       | 0                      | 0     |
| Creation       | 2        | 2               | 37      | 0          | 3          | 0          | 0      | 0            | 95      | 0          | 24      | 0                      | 0     |
| Unclassifiable | 1        | 1               | 17      | 1          | 0          | 0          | 0      | 1            | 7       | 12         | 14      | 2                      | 0     |
| Use            | 23       | 0               | 33      | 7          | 2          | 51         | 90     | 35           | 241     | 0          | 361     | 16                     | 2     |
| TOTAL          | 58       | 4               | 170     | 8          | 8          | 92         | 91     | 78           | 390     | 12         | 435     | 19                     | 2     |

| Categories     | Abstract | Acknowledgments | Article | Case study | Conclusion | Discussion | Figure | Introduction | Methods | References | Results | Supplementary material | Title  |
|----------------|----------|-----------------|---------|------------|------------|------------|--------|--------------|---------|------------|---------|------------------------|--------|
| Background     | 1.7%     | 0.0%            | 25.9%   | 0.0%       | 25.0%      | 37.0%      | 1.1%   | 37.2%        | 4.4%    | 0.0%       | 6.9%    | 5.3%                   | 0.0%   |
| ClinicalTrials | 53.4%    | 25.0%           | 22.9%   | 0.0%       | 12.5%      | 7.6%       | 0.0%   | 16.7%        | 7.7%    | 0.0%       | 0.5%    | 0.0%                   | 0.0%   |
| Compare        | 0.0%     | 0.0%            | 0.0%    | 0.0%       | 0.0%       | 0.0%       | 0.0%   | 0.0%         | 0.0%    | 0.0%       | 0.9%    | 0.0%                   | 0.0%   |
| Creation       | 3.4%     | 50.0%           | 21.8%   | 0.0%       | 37.5%      | 0.0%       | 0.0%   | 0.0%         | 24.4%   | 0.0%       | 5.5%    | 0.0%                   | 0.0%   |
| Unclassifiable | 1.7%     | 25.0%           | 10.0%   | 12.5%      | 0.0%       | 0.0%       | 0.0%   | 1.3%         | 1.8%    | 100.0%     | 3.2%    | 10.5%                  | 0.0%   |
| Use            | 39.7%    | 0.0%            | 19.4%   | 87.5%      | 25.0%      | 55.4%      | 98.9%  | 44.9%        | 61.8%   | 0.0%       | 83.0%   | 84.2%                  | 100.0% |
| TOTAL          | 100.0%   | 100.0%          | 100.0%  | 100.0%     | 100.0%     | 100.0%     | 100.0% | 100.0%       | 100.0%  | 100.0%     | 100.0%  | 100.0%                 | 100.0% |

<p align="center">
  <img width="70%" height="70%" src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/ByCategoriesRepartitionGraphic.PNG">
</p>

<a name="firstresult07"></a>

**With Citations only** :
```
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       0.80      0.70      0.75        53
ClinicalTrials       1.00      0.93      0.96        28
       Compare       1.00      1.00      1.00         1
      Creation       0.82      0.93      0.87        40
Unclassifiable       1.00      0.56      0.72        16
           Use       0.90      0.95      0.93       204

     micro avg       0.89      0.89      0.89       342
     macro avg       0.92      0.84      0.87       342
  weighted avg       0.89      0.89      0.89       342
 0.8888888888888888 	 Logistic Regression 	 4.667 sec
#######################################################
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       1.00      0.17      0.29        53
ClinicalTrials       1.00      0.36      0.53        28
       Compare       0.00      0.00      0.00         1
      Creation       0.94      0.40      0.56        40
Unclassifiable       1.00      0.38      0.55        16
           Use       0.68      1.00      0.81       204

     micro avg       0.72      0.72      0.72       342
     macro avg       0.77      0.38      0.46       342
  weighted avg       0.80      0.72      0.66       342
 0.716374269005848 	 BernoulliNB 	 0.351 sec
#######################################################
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       0.92      0.45      0.61        53
ClinicalTrials       1.00      0.93      0.96        28
       Compare       0.00      0.00      0.00         1
      Creation       0.79      0.95      0.86        40
Unclassifiable       1.00      0.50      0.67        16
           Use       0.85      0.97      0.90       204

     micro avg       0.86      0.86      0.86       342
     macro avg       0.76      0.63      0.67       342
  weighted avg       0.87      0.86      0.84       342
 0.8596491228070176 	 ComplementNB 	 0.056 sec
#######################################################
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       0.89      0.62      0.73        53
ClinicalTrials       1.00      0.89      0.94        28
       Compare       1.00      1.00      1.00         1
      Creation       0.79      0.65      0.71        40
Unclassifiable       1.00      0.56      0.72        16
           Use       0.84      0.98      0.91       204

     micro avg       0.86      0.86      0.86       342
     macro avg       0.92      0.78      0.84       342
  weighted avg       0.87      0.86      0.85       342
 0.8596491228070176 	 GaussianNB 	 0.709 sec
#######################################################
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       1.00      0.06      0.11        53
ClinicalTrials       1.00      0.68      0.81        28
       Compare       0.00      0.00      0.00         1
      Creation       1.00      0.42      0.60        40
Unclassifiable       1.00      0.38      0.55        16
           Use       0.69      1.00      0.81       204

     micro avg       0.73      0.73      0.73       342
     macro avg       0.78      0.42      0.48       342
  weighted avg       0.81      0.73      0.66       342
 0.7280701754385965 	 MultinomialNB 	 0.041 sec
#######################################################
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       1.00      0.43      0.61        53
ClinicalTrials       1.00      0.93      0.96        28
       Compare       1.00      1.00      1.00         1
      Creation       0.92      0.90      0.91        40
Unclassifiable       1.00      0.56      0.72        16
           Use       0.83      0.99      0.90       204

     micro avg       0.87      0.87      0.87       342
     macro avg       0.96      0.80      0.85       342
  weighted avg       0.89      0.87      0.85       342
 0.868421052631579 	 Random Forest 	 3.678 sec
#######################################################
(1025, 5629) X_train shape
(1025,) y_train shape
                precision    recall  f1-score   support

    Background       0.90      0.72      0.80        53
ClinicalTrials       1.00      0.93      0.96        28
       Compare       1.00      1.00      1.00         1
      Creation       0.84      0.93      0.88        40
Unclassifiable       1.00      0.56      0.72        16
           Use       0.91      0.98      0.94       204

     micro avg       0.91      0.91      0.91       342
     macro avg       0.94      0.85      0.88       342
  weighted avg       0.91      0.91      0.91       342
 0.9093567251461988 	 SVM 	 0.378 sec
#######################################################
```
**With Section, SubType, Figure, Pre-Citation, Citation, Post-Citation**
````
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       0.91      0.81      0.86        53
ClinicalTrials       1.00      0.96      0.98        28
       Compare       1.00      1.00      1.00         1
      Creation       0.84      0.90      0.87        40
Unclassifiable       1.00      0.69      0.81        16
           Use       0.93      0.97      0.95       204

     micro avg       0.92      0.92      0.92       342
     macro avg       0.95      0.89      0.91       342
  weighted avg       0.93      0.92      0.92       342
 0.9239766081871345 	 Logistic Regression 	 82.163 sec
#######################################################
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       1.00      0.15      0.26        53
ClinicalTrials       1.00      0.29      0.44        28
       Compare       0.00      0.00      0.00         1
      Creation       1.00      0.23      0.37        40
Unclassifiable       1.00      0.38      0.55        16
           Use       0.66      1.00      0.79       204

     micro avg       0.69      0.69      0.69       342
     macro avg       0.78      0.34      0.40       342
  weighted avg       0.79      0.69      0.62       342
 0.6871345029239766 	 BernoulliNB 	 0.888 sec
#######################################################
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       1.00      0.23      0.37        53
ClinicalTrials       1.00      0.64      0.78        28
       Compare       0.00      0.00      0.00         1
      Creation       1.00      0.50      0.67        40
Unclassifiable       1.00      0.50      0.67        16
           Use       0.72      1.00      0.84       204

     micro avg       0.77      0.77      0.77       342
     macro avg       0.79      0.48      0.55       342
  weighted avg       0.83      0.77      0.73       342
 0.7660818713450293 	 ComplementNB 	 0.169 sec
#######################################################
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       1.00      0.53      0.69        53
ClinicalTrials       1.00      0.86      0.92        28
       Compare       1.00      1.00      1.00         1
      Creation       0.92      0.57      0.71        40
Unclassifiable       1.00      0.50      0.67        16
           Use       0.80      1.00      0.89       204

     micro avg       0.84      0.84      0.84       342
     macro avg       0.95      0.74      0.81       342
  weighted avg       0.87      0.84      0.83       342
 0.8421052631578947 	 GaussianNB 	 3.153 sec
#######################################################
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       0.00      0.00      0.00        53
ClinicalTrials       0.00      0.00      0.00        28
       Compare       0.00      0.00      0.00         1
      Creation       0.00      0.00      0.00        40
Unclassifiable       0.00      0.00      0.00        16
           Use       0.60      1.00      0.75       204

     micro avg       0.60      0.60      0.60       342
     macro avg       0.10      0.17      0.12       342
  weighted avg       0.36      0.60      0.45       342
 0.5964912280701754 	 MultinomialNB 	 0.117 sec
#######################################################
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       1.00      0.42      0.59        53
ClinicalTrials       0.96      0.96      0.96        28
       Compare       1.00      1.00      1.00         1
      Creation       0.97      0.93      0.95        40
Unclassifiable       1.00      0.50      0.67        16
           Use       0.83      1.00      0.91       204

     micro avg       0.87      0.87      0.87       342
     macro avg       0.96      0.80      0.85       342
  weighted avg       0.89      0.87      0.86       342
 0.8742690058479532 	 Random Forest 	 6.673 sec
#######################################################
(1025, 15674)
(1025,)
                precision    recall  f1-score   support

    Background       0.94      0.60      0.74        53
ClinicalTrials       1.00      0.96      0.98        28
       Compare       1.00      1.00      1.00         1
      Creation       0.88      0.93      0.90        40
Unclassifiable       1.00      0.62      0.77        16
           Use       0.89      0.99      0.94       204

     micro avg       0.90      0.90      0.90       342
     macro avg       0.95      0.85      0.89       342
  weighted avg       0.91      0.90      0.90       342
 0.9035087719298246 	 SVM 	 11.238 sec
#######################################################
````

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week8"></a>
### :date: Week 8 - 22/04-28/04 - ML models :


So in the end we have a set of ***1187 citations from 400 papers OpenAccess and containing data citations (~ 3 interesting citations/paper)***.

For a little reminder the **AnnotationAPI found ~ 2 000 citations** that pass the control tests and with the same exact name already found by AnnotationAPI in the same paper there is **almost ~ 4 000 times where the Accession number** is write.

To reach 1187 citations we remove from the citations mined by AnnotationAPI citations that are **shorter than 25 characters**, because most of the times those citations are titles or take place in table or in References, we also remove citations that are **longer than 500 characters**, because **we remove tables** (we decide to not take those citations) but sometimes the AnnotationAPI let pass a table and say it's a sentence, to avoid those citation we've set a limit (**shorter-longer-table citations represent ~600 citations**). We also decide to remove *ClinicalTrials* & *Unclassifiable* **~200 citations**.

:warning: ***There is a problem in cross-validation indeed the compare category seems to "light" to be well represented in those tests***.


I've add stemming apporach, N-gram apporach, Lemma approach (unfortunatly results are corrupted because of a bug)


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Month3"></a>
<a name="Week9"></a>
### :date: Week 9 - 29/04-05/05- Results & Analysis for ML & DL first models :

Fit on a model already fit will delete everything and start from 0.

Firt result are provide from tables below and also we had a suspicion about a feature that could help the model to be better, indeed we thought that the number of paper citations in data citations could be a hint, so we add a column *"NbPaperCitation"* in the dataset (which can be found [here](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/Datasetnb.csv)).

In the end after ML & DL analysis we discover that it bring more noise than help as we can see in those files : [ResultMLnb](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/ResultMLnb.csv) and [ResultDLnb](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/ResultDLnb.csv).

- ***Machine learning model*** [file](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/ResultML1.csv) :

See analysis in the file : [ResultML1](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/ResultML1.xlsx)

- ***Deep learning model*** [file](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/ResultDL1.csv) :

<p align="center">
  <img src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/Approaches-f1scoresDL.PNG">
</p>

<p align="center">
  <img src="https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/Logbook%20%26%20Notes/Analysis/NgramwithDL.PNG">
</p>


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week10"></a>
### :date: Week 10 - 06/04-12/05- Embedding& Cross-validation-score:

This week I worked on Embedding and cross-validation score:

- *Embedding*
  * It seems like the embedding approach work really fine and give pretty optimistic results

- *Cross-validation-score*
  * We decide to use the k-fold function of scikit-learn but there is 2 problems the first one is the importance of the compare class indeed this class as it contains only 4 samples we can't go over 4-fold validationas it take at least 1 to test, so 5-fold and more are not possible. But there is also another problem, it seems that when we use this cross-validation score we run to really much tensorflows OOM errors, so that's an error we can't understand because it's just a loop that recreate the model and test it by the kfold methods. 

It seems those errors came from keras, indeed generating tensors/models in a loop doesn't overwrite previous ones so there is a need to clean the session from tensors and models at each turn of the loop. 

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week11"></a>
### :date: Week 11 - 13/05-19/05- TensorBoard & LSTM :

- *General optimization* of all precedent NN, ML and embedding models
  * Indeed I have to correct my implementation of embedding, ngram, lemmatization, stemming, tokenizer etc..
  So the ngram and embedding are quite similar. But instead of tokenize by bag of words, embedding create a semantic sense so for each word there its "description" but also the "description" of words that are close of this word. In the end embedding represent word througha set of dimension (here from Glove we study 50/100/200/300 dimensions).
  
- *TensorBoard* 
  * It's a tool to visualize what is going on through the trainning, so we can follow the loss, accuracy, validation loss and validation accuracy. It also provide a graph of the network created.

[Paper on LSTM for sentiment analysis](https://www.aclweb.org/anthology/O18-1021)

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week12"></a>
### :date: Week 12 - 20/05-26/05 - LSTM & Embedding & New data extraction :

This week we discover that our dataset dosen't work with LSTM network, we have some ideas why there is a problem here, indeed in our dataset there is some "repetitions" for example it's possible that in a sentence there is multiple different data citations so it result in "duplicates" in our dataset. Our solution is to extract another set of citations and annotate those to complete our dataset and make it bigger.

To make sure that the problem doesn't come from our model we run the model on another dataset (***[SAR14](https://github.com/daiquocnguyen/SAR14)*** from Dai Quoc Nguyen and Dat Quoc Nguyen and Thanh Vu and Son Bao Pham)

In the end we have 800 papers Open Access with data citations (5000 data citations from Annotation API cleaning those (removing shortest, longest, ClinicalTrials..) we have almost 1500 data citations :warning:)

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week13"></a>
### :date: Week 13- 28/05-02/06 - Final model building step :

- *Papers & Blogs* :
  * [Deep learning facilitates rapid classification of human and veterinary clinical
narratives](#Dplearn13) ***Preprint***
  * [Convolutional neural network : Deep learning-based classification of building quality problems](#CNN13)
  
This week we finally reach the end of building model, overpassing all type of errors and checking if results are corrects even if it's not what we expected. We also discover that CNN models gives really good results, indeed regarding the last paper about text classification thanks to machine learning and deep leanring techniques gives us a really good overview of what's going on.
So it has been proved that today CNNs and SVMs are some of the best models for text classification.  

<a name="Dplearn13"></a>
***[Deep learning facilitates rapid classification of human and veterinary clinical
narratives](https://www.biorxiv.org/content/10.1101/429720v2)*** Arturo Lopez Pineda et al.

- :interrobang: ***Problems*** : 
  * Needs to assign clinical codes to patient from diagnoses but it takes a lot of time and result in 60-80% of successfully assign codes.
  
- :heavy_check_mark: ***Solutions*** : 
  * Previous studies *Machine Learning* ( 
    1) Koopman B, Karimi S, Nguyen A, et al. Automatic classification of diseases from free-text death certificates for real-time surveillance. BMC Med Inform Decis Mak 2015;15:53. doi:10.1186/s12911-015-0174-2
    2) Berndorfer S, Henriksson A. Automated Diagnosis Coding with Combined Text Representations. Stud Health Technol Inform 2017;235:201–5.
    3) Anholt RM, Berezowski J, Jamal I, et al. Mining free-text medical records for companion animal enteric syndrome surveillance. Preventive Veterinary Medicine 2014;113:417–22. doi:10.1016/j.prevetmed.2014.01.017
    4) Wang Y, Sohn S, Liu S, et al. A clinical text classification paradigm using weak supervision and deep representation. BMC Med Inform Decis Mak 2019;19:1. doi:10.1186/s12911-018-0723-6)
    The first three ones show high classification accuracy with ML models for human(1,2) and veterinary(3) text narratives for diseases well represented in training set, the last ones (4) show successfuly classification for clinical narratives with Decision Trees, Random Forests and SVMs.
    
   * Previous studies *Deep Learning* (
      1) Weng W-H, Wagholikar KB, McCray AT, et al. Medical subdomain classification of clinical notes using a machine learning-based natural language processing approach. BMC Med Inform Decis Mak 2017;17:155. doi:10.1186/s12911-017-0556-8 
      2) See others in the paper)
   * They used LSTM and RNN (using GloVe for word embedding). In the end LSTM gives best results but it with a f1-score (weighted) with a max at 0.76 but a min of 0.28)

- :triangular_flag_on_post: ***Remaining Problems*** : 
  * Not enough data
  * Not excellent results

<a name="CNN13"></a>
***[Convolutional neural network : Deep learning-based classification of building quality problems](https://www.sciencedirect.com/science/article/pii/S1474034618301538)*** B.Zhong et al.
  
- :interrobang: ***Problems*** : 
  * Classification of building quality complaints(BQCs) result in time consuming and error prone.

- :heavy_check_mark: ***Solutions*** : 
  * Previous studies *Machine Learning* ( 
    1) Z.D. Lu, H. Li, Recent progress in deep learning for NLP, in: Conference of the North American Chapter of the Association for Computational Linguistics: Tutorial, 2016, pp. 11–13.
    2) Y. Kim, Convolutional neural networks for sentence classification, in: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Eprint Arxiv, 2014, pp. 1746–1751. (Sentence levels with only 1 convolution layer)
    3) See others in the paper p.47)
  * Here they use several model to compare the results (CNN, Bayes, SVM) and it seems that CNNs gives the best result following by SVMs and then Bayes models.

- :triangular_flag_on_post: ***Remaining Problems*** : 
  * They want to explore more features
  * Improve f1-score (with transfer learning)

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)


<a name="Week14"></a>
### :date: Week 14- 03/06-10/06 - Finishing building models & Optimisation :

Looking at the current results I think that the "*Background*" category is really tricky, indeed even for us it's difficult to say that a data citation could be a *Background* one. But I'm conviced that sometimes there is data citation that are made in papers but scientists doesn't really use those data eg.

*A2M is an evolutionarily conserved element of the innate immune system and a non-specific protease inhibitor involved in host defense, and it has been revealed that A2M is relative to immunity in L. vannamei [65]. **The F11 gene (GenBank: AFW98990.1) was reported to play a role in immunity [1]. Recent studies revealed the importance of KLKB1 in shrimp immune response, particularly towards protect animals from the microbial pathogens [66].** Aquatic animals metabolize foreign toxicity of chemicals mainly by oxidation, reduction, hydrolysis and conjugation reactions catalyzed by various enzymes, and the metabolic activation is primarily catalyzed by the cytochrome P450-dependent oxygenase system in the endoplasmic reticulum [67].*

&nbsp;&nbsp;&nbsp;&nbsp;from Discussion of [Transcriptome Analysis of the Hepatopancreas in the Pacific White Shrimp (Litopenaeus vannamei) under Acute Ammonia Stress. (PMID:27760162 PMCID:PMC5070816)](http://europepmc.org/articles/PMC5070816?fromSearch=singleResult&fromQuery=PMC5070816)

Here we couldn't say that authors used the F11 gene, we supposed it as they talk about it but it could be just an hypothesis and not a case where they really use this gene. Here it's a problem of level, indeed we work only on sentence level (1 before citation, *Citation*, 2 after citation) but the information that they use actually this gene in this study could be in another part of the paper, so it could be great to have an overview of the whole paper.

One other possibility is to look at the citation in the paper corresponding to this specific citation, if this citation is a *Background* one but two lines before the authors told they use actually those data so it could consider as a *Use* one. And even if in the beginning or at the end of the paper authors says that they "create" those data so it could be consider as a *Create* one. So my thought is those categories have a sort of "level of priority" *Background* < *Use* < *Creation*.

After building and running all models we found a classification of all models used until there. It seems that Logistic Regression is the best model for this task followed by SVMs then CNN and after a simple NN. As we can see in this [file](https://github.com/0AlphaZero0/Sentiment-Analysis-Data-Citation-EuropePMC/blob/master/MLAnalysis/AllResult.xlsx) So we decide to keep those to une them and see if there is one that really is great comparing to other selected models.

During this week we optmized thanks to Grid search, 4 models (Logistic Regression, SVM, CNN and also "Dplearn" model) as thos four models were the best during the training part.

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week15"></a>
### :date: Week 15- 03/06-10/06 - :


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

______________________________________________________________________

<a name="bibliography01"></a>
## :bookmark_tabs: Bibliography :

- [Deep Learning, NLP, Representation](#Deep1)
- [SciLite](#SciLite1)
- [Database citation in full text biomedical papers](#Database1)
- [EuropePMC](#EuropePMC1)
- [Sentiment analysis of scientific citation](#Sentiment1)
- [Measuring the Evolution of a Scientific Field through Citation Frames](#Measuring2)
- [New Features for Sentiment Analysis: Do Sentences Matter?](#NewFeatures3)
- [Sentiment Symposium Tutorial](http://sentiment.christopherpotts.net/index.html)
- [AUTOLEX: An Automatic Lexicon Builder for Minority Languages Using an Open Corpus](#AUTOLEX3)
- [Data Citation Synthesis Group: Joint Declaration of Data Citation Principles. Martone M. (ed.) San Diego CA: FORCE11; 2014](https://doi.org/10.25490/a97f-egyk)
- [Achieving human and machine accessibility of cited data in scholarly publications](#Achieving4)

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

______________________________________________________________________

<a name="index01"></a>
## :bookmark_tabs: Index :

**ACL** : digital archive which contains conference and journal papers in NLP and CL

**AdaBoost.HM** : meta-algorithm of boosting

**BoosTexter** : software that use machine learning to classify text data --> :warning: it's a general tool, it is not accurate for scientific texts.

**Boosting** : It's a part of machine learning algorithm that improves algorithm performances thanks to binary classifier.

**CL** : Computational linguistics

**CRF** : Champ aléatoire conditionnel (classes de modèles stats utilisées pour des données séquentielles	 (langage)

**DB** : Database/s

**ENA** : European Nucleotide Archive

**EPMC** : Europe PMC / Europe PubMed Central

**EPO(DOCDB)** : European Patent Office ; Base de données documentaire (biblio, abrégés, citations,	 DOCDB = ensemble de brevets       nécessaire pour une invention)


**IBk** : Algorithm of machine learning ≈ K nearest algorithm

**IPC** : International Patent Classification hierarchical system of language independent symbols for the	 classification of patents and utility models according to the different areas of technology to	 which they pertain (classification des DOCDB-ensemble de brevets- selon certains paramètres 	tel que leur domaines)

**Lemma** : In psycholinguistics, a lemma (plural lemmas or lemmata) is an abstract conceptual form of a word that has been mentally selected for utterance in the early stages of speech production.[1] A lemma represents a specific meaning but does not have any specific sounds that are attached to it.

**Lexicon** : The vocabulary uses by one person or one field (like scientific lexicon)

**MPQA** : paper databank

**MRF** : Markov Random Field a model of non oriented graphs

**N-grams** : Computationnal linguistics, sequence of *n-items* from a text or speech

**NLP** : Neuro-linguistic programming

**OA** : Open Access

**ORCID** : code alphanumérique non propriétaire, qui permet d'identifier de manière unique les chercheurs et auteurs de contributions académiques et scientifiques

**Perceptron** : Supervised learning algorithm with binary classifiers

**PDB** : Protein Data Bank

**PMI** : Pointwise Mutual Information (info mutuelle ponctuelle  indice d’information partagé entre 2 mots par exemple 2 mots positifs l’indice serait élevé)

**PMID** : ID for articles with only an abstract

**PMCID** : ID for a full-txt article of Europe PMC

**Precision** : positive predictive values  fraction of relevant instances among the retrieved instances fraction de vrai posisitf / (vrai positif + faux positif) le nombre d’identifications correctes sur toutes les identifications réalisées ([see below](#PrecisionRecall01))

**RDF** : Resource Description Framework modèle de graphe destiné à décrire de façon formelle les ressources Web, fonctionne avec des ensembles de triplets (sujet : ressource à décrire, prédicats : type de propriété applicable à la ressource, objets : donnée ou ressource)

**Recall** : the fraction of relevant instances that have been retrieved over the total amount of relevant instances  fraction de vrais positifs/(vrais positifs+faux negatifs) le nombre d’identification correct parmi l’ensemble des corrects ([see below](#PrecisionRecall01))

**Recursive neural network / Models / Matrix Vector RNN** : ???

**RefSeq** : NCBI Reference Sequence Database

**RESTful API** : an application program interface(API) that uses HTTP requests to GET,PUT,POST, and DELETE data.

**RNN** : Recursive Neural Network

**SureChEMBL** : Biology DataBank

**Synset** : Each set in WordNet corresponds to a word sense, not a word string. Each set represents a unique concept and members of each set, each of which is a sense, are synonyms of each other. These sets are also called synsets. (It’s like a pack of word meaning the same thing.)


**t-SNE** algortithm : Proximity between 2 points ==> usefull with a lot of features, it's easier to see points that are really close        (:warning: more than 3 dimensions)

**STATA** : Outil statistique

**USPTO** : US Patent and Trademark Office

**WIPO** : World Intellectual Property Organisation

**Word/SentiWordNet** : 2 lexicons, SentiWordNet is made for opinion mining

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)


______________________________________________________________________

<a name="attachments01"></a>
## :bookmark: Attachments :

<a name="PrecisionRecall01"></a>
<p align="center">
  <img src="https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/PrecisionRecall.png">
</p>

<p align="center">
Precision & Recall
</p>

So to make a good analysis, it's necessary to have a high precision & recall

<a name="POStag01"></a>
<p align="center">
  <img src="https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/POStag.png">
</p>

<p align="center">
POS-tag
</p>

<a name="dependencytrees3"></a>
<p align="center">
  <img src="https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/Wearetryingtounderstandthedifference.jpg">
</p>

<p align="center">
Dependency trees
</p>

These trees illustrate two possible ways to render the dependency and phrase structure relations (see below). This dependency tree is an "ordered" tree, i.e. it reflects actual word order. Many dependency trees abstract away from linear order and focus just on hierarchical order, which means they do not show actual word order. This constituency (= phrase structure) tree follows the conventions of bare phrase structure (BPS), whereby the words themselves are employed as the node labels.

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)
