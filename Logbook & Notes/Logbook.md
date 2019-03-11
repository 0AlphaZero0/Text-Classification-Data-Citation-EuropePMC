<h1 align="center">SACSP</h1>
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

<a name="11/03"></a>
### 11/03 - Defintions of planinng, needs and requests :

Meetings :
 - Every Tuesday at 11:00 with Xiao Yang.
 - Every 4 weeks at 11:00 with Xiao Yang & Aravind Venkatesan & Johanna McEntyre.
 
Needs & requests :
 - Logbook needed.
 - End of July : 
  - Good draft of the internship report.
  - Presentation of the work done


______________________________________________________________________
<a name="Week1"></a>
### Week 1 - 01/03-10/03 :

#### :diamond_shape_with_a_dot_inside: Deep Learning, NLP, Representation :

http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

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
 
#### :diamond_shape_with_a_dot_inside: SciLite :

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

#### :diamond_shape_with_a_dot_inside: [EuropePMC](http://europepmc.org/) (EPMC/Europe PubMed Central):

PMID = ID for non full text papers (Abstract only) / PMCID = ID for full text papers
 
Pipeline EPMC's : text mining to extract accession references
perform with [STATA](https://www.stata.com/) (statistical tool)
![](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/EuropePMCschema1.PNG)

Features : Accession ID; Deposition ID; Deposition Year; First Public Year; PMID publication year; citation year; citation

#### :diamond_shape_with_a_dot_inside: Database citation in full text biomedical papers :

*EPMC* :heavy_plus_sign: *NLM* (National Library of Medicine) :arrow_forward: .XML thanks to OCR

Identification oof citation thanks to WhatizitANA pipeline

Problems





______________________________________________________________________

<a name="index"></a>
### Index :
