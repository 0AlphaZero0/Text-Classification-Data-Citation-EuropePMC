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
- [11/03 - Defintions of planinng, needs and requests](#11/03)
- [12/03 - Questions & path set up](#12/03)
- [19/03 - Dataset conception](#19/03)
- [25/03 - XML Talks / Data creation](#25/03)
- [02/04 - Summary of March](#02/04)

______________________________________________________________________
### :green_book: Notes

- [Month 1 - 01/03-31/03](#Month1)
  * [Week 1 - 01/03-10/03](#Week1)
  * [Week 2 - 11/03-17/03](#Week2)
  * [Week 3 - 18/03-24/03](#Week3)
  * [Week 4 - 25/03-31/03](#Week4)
- [Month 2 - 01/04-30/04](#Month2)
  * [Week 5 - 01/04-07/04](#Week5)
  * [Week 6 - 08/04-14/04](#Week6)
  * [Week 7 - 15/04-21/04](#Week7)
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

______________________________________________________________________

## :green_book: Notes

<a name="Month1"></a>
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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

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

<a name="Teufeljson"></a>
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

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week3"></a>
### :date: Week 3 - 18/03-24/03 :

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
### :date: Week 4 - 25/03-31/03 :

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
### :date: Week 5 - 01/04-07/04 :

I've try to improve a kind of **pipeline to create dataset for the future analysis**, for this First I extract papers that are open access and contains Accession Numbers so there is a full-txt and an Accession-Numbers file in XML format from the annotation API and RESTful API. Then I extract a PMCID list that it used to "sentencized" full-txt XML files. Then I use a little script to repare result files, indeed the splitter sometimes made mistakes, And it's kind of easy to fix some of these. Then I extract citation from those repared files and it result in a file with all sentences containing all citations. But we're focusing on citations that are currently mine by the annotation API.

To create a good dataset we decide to take the **section** and the **subtype** of the citation eg. section= *Results* or *Methods* and SubType= *ENA* or *PDBe*.

We also decide to **remove citations** that have less than 25 characters and more than 500 characters, indeed those with less than 25 are most of the time title like *INTRODUCTION* or *Suplementary-material*.
Thanks to a little analysis we fix those two limits indeed most of the length of citations and context sentences are between 25 and 800. But those which are mined start from 1 to 30000 characters.

And I've also notice that citations are mostly at the end of a paragraph unlike the beginning of it.

It could be great to add a feature **Figure** that can be set to *True* or *False* or reaplce the section feature (Abstract, Methods etc. by *Figure* when the citation take place in a caption's figure.

At the end I've start to annotate ~1000 citations with categories : **Background, Use, ClinicalTrials, Creation **

[:top::top::top::top::top::top::top::top::top::top::top::top::top::top:Go to the top:top::top::top::top::top::top::top::top::top::top::top::top::top::top::top:](#top)

<a name="Week6"></a>
### :date: Week 6 - 08/04-14/04 :

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
### :date: Week 7 - 15/04-21/04 :

#### :bar_chart: Analysis :

The final dataset or the pipeline after annotation will look like :

| PMCID      | AccessionNb | Section | SubType | Figure | Categories | Pre-citation                                                                                                                                            | Citation                                                                                                                                                                                                                                                                                                                 | Post-citation                                                                                                                                                                                                                                                                                                                                                                                                   |
|------------|-------------|---------|---------|--------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PMC1940057 | AM282973    | Results | ENA     | False  | Use        | This identity can be explained by the high-stringency PCR conditions and the low degree of degeneration of the two used primers.                        | Frame analysis of the nucleotide sequence (696 bp, accession no. AM282973) revealed the presence of a unique internal open reading frame.                                                                                                                                                                                | The alignment of the corresponding amino acid sequence (aa) with all available protein sequences using the Gapped BLAST and PSI-BLAST program [24] showed as expected important aa identities with different NRPS adenylation domains. This identity reaches 50% and 48% with the adenylation domains of the pristinamycin I (PI) synthetase 2, and actinomycin (ACM) synthetase III, respectively (Figure 1).  |
| PMC2154357 | 1BL8        | Methods | PDBe    | False  | Background | All values are reported in Table I together with previous results on KcsA (Bernèche and Roux, 2001; Noskov et al., 2004).                               | We estimate that the accuracy and overall significance of the calculated free energies is roughly on the order of 1 kcal/mol, based on the difference between the computations with 1K4C (Noskov et al., 2004) and 1BL8 (Bernèche and Roux, 2001) and the comparison between the CHARMM PARAM27 and AMBER force fields.  | For the simulations with the AMBER force field, the systems were reequilibriated for 1.5 ns before starting the free energy computations. The hydration number of K+ and Na+ in the various binding sites were computed from the average of 800 ps of MD for each configuration; statistical error was estimated by comparing block averages.                                                                   |
| PMC2216687 | Q9XVV3      | Results | UniProt | False  | Background | We first explored this concept for PPI pairs from different species and have observed evidence of this conservation of function between the PPI pairs.  | For example, in C. elegans, nhr-67 [Swiss-Prot: Q9XVV3] and daf-21 [Swiss-Prot: Q18688] have been shown to interact [27], whereas in human ESR1 [Swiss-Prot: P03372] and HSP90AA1 [Swiss-Prot: P07900] are also known to interact [28].                                                                                  | Both PPI pairs contain a common domain interaction pattern, (PF00105)-(PF02518, PF00183), where ‘-’ denotes interaction and the parentheses denote modular domains. PF00105 is described by Pfam [29] as the zinc finger, C4 type domain, and PF02518 and PF00183 refer to HATPase_c and HSP90 domains, respectively.                                                                                           |
| PMC2216687 | Q18688      | Results | UniProt | False  | Background | We first explored this concept for PPI pairs from different species and have observed evidence of this conservation of function between the PPI pairs.  | For example, in C. elegans, nhr-67 [Swiss-Prot: Q9XVV3] and daf-21 [Swiss-Prot: Q18688] have been shown to interact [27], whereas in human ESR1 [Swiss-Prot: P03372] and HSP90AA1 [Swiss-Prot: P07900] are also known to interact [28].                                                                                  | Both PPI pairs contain a common domain interaction pattern, (PF00105)-(PF02518, PF00183), where ‘-’ denotes interaction and the parentheses denote modular domains. PF00105 is described by Pfam [29] as the zinc finger, C4 type domain, and PF02518 and PF00183 refer to HATPase_c and HSP90 domains, respectively.                                                                                           |

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
