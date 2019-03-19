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
<p align="center">Sentiment Analysis on Citations in Scientific Papers in Europe Pub Med Central</p>

______________________________________________________________________

### :closed_book: Meetings

- [05/03 - CambMetrics](#05/03)
- [11/03 - Defintions of planinng, needs and requests](#11/03)
- [12/03 - Questions & path set up](#12/03)

______________________________________________________________________
### :green_book: Notes

- [Week 1 - 01/03-10/03](#Week1)
- [Week 2 - 11/03-17/03](#Week2)
- [Week 3 - 18/03-24/03](#Week3)

______________________________________________________________________

### :bookmark_tabs: Index

[Index](#index01)

______________________________________________________________________

### :bookmark: Attachments

[Attachments](#attachments01)

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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

______________________________________________________________________

## :green_book: Notes

<a name="Week1"></a>
### :date: Week 1 - 01/03-10/03 :

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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week2"></a>
### :date: Week 2 - 11/03-17/03 :

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

<a name="Teufel-json"></a>
In fact there is a lack of data to this problem only *"Measuring the Evolution of a Scientific Field through Citation Frames D.Jurgens et al."* proposed a really good dataset [Teufeljson](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Datasets/teufel-json.tar.gz) thanks to ACL Anthology Network Corpus. But there is some problems here, this dataset is quiet heavy due to information duplication. Also, this dataset isn't annotated with polarity it's only with 6 classes that I described after.

<a name="CitationSentimentCorpus"></a>
There is also the dataset : [Citation sentiment corpus](https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Datasets/Citation_sentiment_corpus.zip), this one look really good, indeed it use objective/positive/negative classes, but one problem is that citation's contexts are in html files. and aren't annotated directly.


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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week3"></a>
### :date: Week 3 - 18/03-24/03 :

- *Papers & Blogs* :
  * [New Features for Sentiment Analysis: Do Sentences Matter?](#NewFeatures3)
  * [Sentiment Symposium Tutorial](http://sentiment.christopherpotts.net/index.html)

There is some datasets that already exists for sentiment analysis like :
 * [***Sentiment140***](http://help.sentiment140.com/for-students/), this dataset is really popular because it contains 160,000 tweets collected via Twitter API (unfortunately emoticons are pre-removed, there is actually some research about sentiment analysis using emoticons and they provide a lot of good results), unfortunately this dataset can't be used here, because we're focusing on scientific paper and not on tweets.
 * [***Stanford Sentiment Treebank***](https://nlp.stanford.edu/sentiment/code.html), this dataset like the precedent one is not focusing on scientific papers (entertainment review website) so we're not using it.
 * [***Multidomain sentiment analysis dataset***](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/), this dataset is a multi domain-one (Amazon reviews), but as I can read in many papers, it's actually quite hard to obtain good results with multi-domain, so we're focusing only in scientific papers.
 * [***Paper Reviews Data Set***](https://archive.ics.uci.edu/ml/datasets/Paper+Reviews), this dataset could be great because it contains scientific papers reviews, but it's only reviews and only focusing on computing and informatic, the goal of this project is to apply sentiment analysis on biomedical scientific papers, so this one is not really adapted for the task.
 * [***Awais Athar - Citation Sentiment Corpus***](http://cl.awaisathar.com/citation-sentiment-corpus/), this dataset that I've find in the thesis of Awais Athar, is quite good but as said [earlier](#CitationSentimentCorpus), it's maybe not the perfect one to use here.
 * [***Teufel-json***](http://jurgens.people.si.umich.edu/citation-function/), like the one before I've discussed about it [before](#Teufeljson) and it may not be the good one to use here.

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


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

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
So to make a good analysis, it's necessary to have a high precision & recall

<a name="POStag01"></a>
<p align="center">
  <img src="https://github.com/0AlphaZero0/Sentiment-Analysis-EuropePMC/blob/master/Logbook%20%26%20Notes/POStag.png">
</p>


[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Got to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)
