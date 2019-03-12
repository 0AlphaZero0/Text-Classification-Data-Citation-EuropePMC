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
<p align="center">Sentiment Analysis on Citations in Scientific Papers in Europe Pub Med Central</p>

______________________________________________________________________

### Meetings

- [11/03 - Defintions of planinng, needs and requests](#11/03)

______________________________________________________________________
### Notes

- [Week 1 - 01/03-10/03](#Week1)

______________________________________________________________________

### Index

[Index](#index)

______________________________________________________________________

<a name="11/03"></a>
### 11/03 - Defintions of planinng, needs and requests :

Meetings :
 - Every Tuesday at 11:00 with Xiao Yang.
 - Every 4 weeks at 11:00 with Xiao Yang & Aravind Venkatesan & Johanna McEntyre.
 
Needs & requests :
 - Logbook needed.
 
 End of July : 
  - Good draft of the internship report.
  - Presentation of the work done


______________________________________________________________________
<a name="Week1"></a>
### Week 1 - 01/03-10/03 :

#### :diamond_shape_with_a_dot_inside: [Deep Learning, NLP, Representation](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) :

 - **Perceptron** : Supervised learning algorithm with binary classifiers
 - **t-SNE** algortithm : Proximity between 2 points ==> usefull with a lot of features, it's easier to see points that are really close        (:warning: more than 3 dimensions)
 - **NLP** : Neuro-linguistic programming
 - **N-grams** : Computationnal linguistics, sequence of *n-items* from a text or speech
 - **Recursive neural network / Models / Matrix Vector RNN** : ???
 
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
 
#### :diamond_shape_with_a_dot_inside: [SciLite](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/SciLite%20a%20platform%20for%20displaying%20text-mined%20annotations_A.Venkatesan_et_al.pdf) :

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

#### :diamond_shape_with_a_dot_inside: [EuropePMC](http://europepmc.org/) (EPMC/Europe PubMed Central) [PDF](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Database%20citation%20in%20supplementary%20data%20linked%20to_S.Kafkas_et_al.pdf):

PMID = ID for non full text papers (Abstract only) / PMCID = ID for full text papers
 
Pipeline EPMC's : text mining to extract accession references
perform with [STATA](https://www.stata.com/) (statistical tool)
![](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/EuropePMCschema1.PNG)

Features : Accession ID; Deposition ID; Deposition Year; First Public Year; PMID publication year; citation year; citation

#### :diamond_shape_with_a_dot_inside: [Database citation in full text biomedical papers](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Database%20Citation%20in%20Full%20Text%20Biomedical%20Articles_S.Kafkas_et_al.pdf) :

*EPMC* :heavy_plus_sign: *NLM* (National Library of Medicine) :arrow_forward: .XML thanks to OCR

Identification oof citation thanks to *WhatizitANA* pipeline

***Problems*** : 
 - Footnotes
 - *ENA* does not include RefSeq
 - Sometimes the pipeline partially identifies accession numbers
 - other errors from wrong assignation of ID like *GenBank* instead of *UniProt*

#### :diamond_shape_with_a_dot_inside: [Sentiment analysis of scientific citation](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Bibliography/Sentiment%20analysis%20of%20scientific%20citations_A.Athar_et_al.pdf) :

***Problems known*** :
 - Many citations in the same sentence
 - Sentiment in citation are often hidden
 - Citations are often neutral (description of a method) ==> Subjective / Objective
 - Much variation between scientific text and other genres on lexical terms (lexicon, eg : "surprising movie" / "surprising results")
 - Sentiment could be in citation sentence but also arround (like in the next 3 ones)
 
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







______________________________________________________________________

<a name="index"></a>
### Index :

**ACL** : digital archive which contains conference and journal papers in NLP and CL

**AdaBoost.HM** : meta-algorithm of boosting

**BoosTexter** : software that use machine learning to classify text data --> :warning: it's a general tool, it is not accurate for scientific texts.

**Boosting** : It's a part of machine learning algorithm that improves algorithm performances thanks to binary classifier.

**CL** : Computational linguistics

**CRF** : Champ aléatoire conditionnel (classes de modèles stats utilisées pour des données séquentielles	 (langage)

**ENA** : European Nucleotide Archive

**EPMC** : Europe PMC / Europe PubMed Central

**EPO(DOCDB)** : European Patent Office ; Base de données documentaire (biblio, abrégés, citations,	 DOCDB = ensemble de brevets       nécessaire pour une invention)


**IBk** : Algorithm of machine learning ≈ K nearest algorithm

**IPC** : International Patent Classification hierarchical system of language independent symbols for the	 classification of patents and utility models according to the different areas of technology to	 which they pertain (classification des DOCDB-ensemble de brevets- selon certains paramètres 	tel que leur domaines)

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

**Précision** : positive predictive values  fraction of relevant instances among the retrieved instances fraction de vrai posisitf / (vrai positif + faux positif) le nombre d’identifications correctes sur toutes les identifications réalisées  voir figures ci-dessous

**RDF** : Resource Description Framework modèle de graphe destiné à décrire de façon formelle les ressources Web, fonctionne avec des ensembles de triplets (sujet : ressource à décrire, prédicats : type de propriété applicable à la ressource, objets : donnée ou ressource)

**Recall** : the fraction of relevant instances that have been retrieved over the total amount of relevant instances  fraction de vrais positifs/(vrais positifs+faux negatifs) le nombre d’identification correct parmi l’ensemble des corrects  voir figure ci-dessous

**Recursive neural network / Models / Matrix Vector RNN** : ???

**RefSeq** : NCBI Reference Sequence Database

**RESTful API** : an application program interface(API) that uses HTTP requests to GET,PUT,POST, and DELETE data.

**RNN** : Recursive Neural Network

**SureChEMBL** : ???

**Synset** : Each set in WordNet corresponds to a word sense, not a word string. Each set represents a unique concept and members of each set, each of which is a sense, are synonyms of each other. These sets are also called synsets. (It’s like a pack of word meaning the same thing.)


**t-SNE** algortithm : Proximity between 2 points ==> usefull with a lot of features, it's easier to see points that are really close        (:warning: more than 3 dimensions)

**STATA** : Outil statistique

**USPTO** : US Patent and Trademark Office

**WIPO** : World Intellectual Property Organisation

**Word/SentiWordNet** : 2 lexicons, SentiWordNet is made for opinion mining

______________________________________________________________________
![](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/PrecisionRecall.png)
So to make a good analysis, it's necessary to have a high precision & recall
