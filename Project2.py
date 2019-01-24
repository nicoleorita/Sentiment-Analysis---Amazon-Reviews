# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:49:21 2018

@author: Nicole Rita
"""

## Pre-Processing
# The code was runned one time for the Train data and again to the Test data.

import json
import nltk
import pandas as pd
import numpy as np
import re
import random
import sys

#-------------------------------2nd project

df_train_total_2 = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - clean_lemmatize.pkl")
df_test_total_2 = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - test_clean_lemmatize.pkl")

'''
my_list = [('this', 'p'), ('product', 'n'), ('is', 'v'), ('funny', 'a'), ('and', 'p'),('scary', 'a'), ('and', 'p'), ('it', 'p'), ('is', 'v'), ('an', 'p'), ('insanely', 'a'),('great', 'a'), ('movie', 'n'), ('loved', 'v'), ('the', 'p'),('exciting', 'a'), ('ending', 'n')]

my_text = ["I am a product of bags and drops beautiful empty legendary epic makeup cute watch cookies", "baby went to buy me a beautiful set of epic makeup", "legendary so epic", "legendary epic"]
my_cenas = [['hey', 'hello', 'i_am', 'no_dont'], ['limao', 'limon_cenas']]
'''
#-----------------------------------------------DF ADJECTIVE NOUNS----------------------
list_aux = list()
lista_final = list()

lista_end = list()
for text in df_train_total_2.get('reviewText'):
    pos_tagged_text = pos_tag_text(text)
    s = len(pos_tagged_text)
    lista_final = list()
    for i in range(0, len(pos_tagged_text)):
        if pos_tagged_text[i][1] =='a':
            lista_final.append(pos_tagged_text[i][0])
            if i+1<s and pos_tagged_text[i+1][1] == 'n':
                list_aux.append(pos_tagged_text[i][0] + "_" + pos_tagged_text[i+1][0])
                lista_final.append(list_aux)
                list_aux = list()
            if i+2<s and pos_tagged_text[i+2][1] == 'n':
                list_aux.append(pos_tagged_text[i][0] + "_" +pos_tagged_text[i+2][0] )
                lista_final.append(list_aux)
                list_aux = list()
    lista_end.append(lista_final)
    

lnew = list()
for i in lista_end:
    #print('>>>',i)
    fnew=list()
    for j in i:
        #print(j)
        if type(j) == list:
            #j = j[0]
            fnew.append(j[0])
        else:
            fnew.append(j)
    lnew.append(fnew)
    
df_adj_nouns = df_train_total_2
df_adj_nouns['reviewText'] = lnew

join_reviews= [ ' '.join(l) for l in df_adj_nouns['reviewText']]
df_adj_nouns['reviewText'] = join_reviews


# SAVE THE BAG OF WORDS WITH NOUNS
df_adj_nouns.to_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\5 - df_adjective_nouns.pkl")
df_adj_nouns = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\5 - df_adjective_nouns.pkl")

#------------------------------------TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.0, max_df = 1.0,
ngram_range=(1, 1))

feature_matrix = vectorizer.fit_transform(df_adj_nouns).astype(float)
print(feature_matrix)
vectorizer, train_features = build_feature_matrix(documents=df_adj_nouns,feature_type='tfidf',
ngram_range=(1, 1),
min_df=0.0, max_df=1.0)
#-------------------------------------------DF NOUNS, ADJECTIVES, ADVERBS----------------------------------------
from nltk import pos_tag

lista_naa = list()
for l in df_train_total_2.get('reviewText'):
    review = nltk.tokenize.word_tokenize(l)
    reviewNew = nltk.pos_tag(review)
    lista_review = list()
    for i in reviewNew:
        if i[1] == 'NN' or i[1] == 'JJ' or i[1] == 'RB':
            lista_review.append(i[0])
    lista_naa.append(lista_review)
 '''       
        elif i[1] == 'JJ':
            lista_review.append(i[0])
        elif i[1] == 'RB':
            lista_review.append(i[0])
'''

df_adj_nouns_adv = df_train_total_2
df_adj_nouns_adv['reviewText'] = lista_naa

join_reviews= [ ' '.join(l) for l in df_adj_nouns_adv['reviewText']]
df_adj_nouns_adv['reviewText'] = join_reviews


# SAVE THE DATAFRAME WITH ADJECTIVES, NOUNS & ADVERBS
df_adj_nouns_adv.to_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\6 - df_adjective_nouns_adv.pkl")
df_adj_nouns_adv = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\6 - df_adjective_nouns_adv.pkl")



##############################################################################################################
                                                  # BOW #
##############################################################################################################

# The code bellow was used to understand the Bag of Words, but during the report we decided to
# use another way to do this part, because it fits better with SVM

X_train = df_train_total_2["reviewText"]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_train_features = X_train_vectorized.todense()
X_train_vocab = vectorizer.vocabulary_

# Count words to see the most common words in the dataset
def words(text): return re.findall(r'\w+', text.lower())

WORDS_train = []
for text in df_train_total_2.get('reviewText'):
    WORDS_train.extend(words(text))

from collections import Counter

count_words = Counter(WORDS_train)
#check the top 50 words used in all reviews
cenas = Counter(WORDS_train).most_common(20)
#number of unique words in all reviews
len(Counter(WORDS_train))

count_vectorizer = CountVectorizer(stop_words='english')
#words, word_values = get_top_n_words(n_top_words=20, count_vectorizer=count_vectorizer, text_data=reindexed_data)
import matplotlib.pyplot as plt

#--------------------------------------------PLOT TOP 20 WORDS IN TRAINING SET
fig, ax = plt.subplots(figsize=(10,5))
ax.barh( range(len(cenas)), [t[1] for t in cenas] , height = 0.5 , align="center", color='#B2C5D8')
ax.set_yticks(range(len(cenas)))
ax.set_yticklabels(t[0] for t in cenas)
plt.title("Top 20 words in training set")
plt.tight_layout()
plt.show()

#--------------------------------------------------!!!!!!!!!!!!!! TF-IDF
tokenize = lambda doc: doc.lower().split(" ")

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

#in Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
print(sklearn_representation) # RESULT (document, word) tfidf value

#-----------------------------!!!!!!!!!!!

## Bag Of Words treated to Support Vector Machines

# IMPORTS
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools



#TRAIN CORPUS - for bow normal & K-BEST
train_corpus = df_train_total_2['reviewText'].tolist()
train_label = df_train_total_2['overall'].tolist()
#TEST CORPUS - for the three models
test_corpus = df_test_total_2['reviewText'].tolist()
test_label = df_test_total_2['overall'].tolist()

#adjectives and nouns corpus train
train_corpus_adj_nouns = df_adj_nouns['reviewText'].tolist()
train_label_adj_nouns = df_adj_nouns['overall'].tolist()

#adjectives, adverbs and nouns corpus train
train_corpus_adj_nouns_adv = df_adj_nouns_adv['reviewText'].tolist()
train_label_adj_nouns_adv = df_adj_nouns_adv['overall'].tolist()


#STEP 1 - CHOOSE ONE OF THE MODELS TO RUN
#-----------------------------------------------BOW normal
def bow_extractor(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()

vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW adjevtive, adverb, nouns
def bow_extractor_bigrams(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor_bigrams(train_corpus_adj_nouns_adv)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW Adjective & Nouns
def bow_extractor_nouns(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor_nouns(train_corpus_adj_nouns)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW for K-Best
def bow_extractor(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
bow_matrix = np.array(bow_matrix, dtype='float64')

vocab = bow_vectorizer.get_feature_names()

kbest_classifier = SelectKBest(chi2, k=5000)
kbest_features = kbest_classifier.fit_transform(bow_matrix, test_label)

'''
bool_list = kbest_classifier.get_support()
kbest_features_cenas = list()


for bool, feature in zip(bool_list, vocab):
    if bool:
       kbest_features_cenas.append(feature)
kbest_dataframe = pd.DataFrame(kbest_features, columns=kbest_features_cenas)
'''

#--------------------------------------------------------------
#STEP 2 - EQUAL FOR ALL THREE MODELS

def get_metrics(true_labels, predicted_labels):

    print ('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels,
                                               predicted_labels),
                        2))
    print ('Precision:', np.round(
                        metrics.precision_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('Recall:', np.round(
                        metrics.recall_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('F1 Score:', np.round(
                        metrics.f1_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

mnb = MultinomialNB()
svm = LinearSVC(penalty='l2', C=1.0) # THE PARAMETERS FROM HANDOUT

# Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label,
                                           test_features=bow_test_features,
                                           test_labels=test_label)
# SVM with K-Best
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label,
                                           test_features=kbest_features,
                                           test_labels=test_label)


cm = metrics.confusion_matrix(test_label, svm_bow_predictions)
pd.DataFrame(cm, index=range(0,5), columns=range(0,5))


#######################################################################################################
                                         #SENTIMENT LEXICON#
#######################################################################################################
df_test_before_stopwords = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\2 - test total cleaned.pkl")
df_train_before_stopwords = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - train_clean_lemmatize_no_stop.pkl")

df_train_with_label = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\1 - df_train_total_with_label.pkl")



lfinal = []
for t in df_train_before_stopwords.get('reviewText'):
    laux=list()
    for w in t.split():
        #print(w)
        laux.append(w)
    lfinal.append(laux)

df_train_before_stopwords['reviewText'] = lfinal


tuple_tuples = tuple(tuple(txt) for txt in lfinal)
    

import sys
from collections import defaultdict
from operator import itemgetter
from numpy import dot, sqrt, array

def cooccurrence_matrix(corpus):
    """
    Create the co-occurrence matrix.

    Input
    corpus (tuple of tuples) -- tokenized texts

    Output
    d -- a two-dimensional defaultdict mapping word pairs to counts
    """    
    d = defaultdict(lambda : defaultdict(int))
    for text in corpus:
        for i in range(len(text)-1):            
            for j in range(i+1, len(text)):
                w1, w2 = sorted([text[i], text[j]])                
                d[w1][w2] += 1
    return d

def get_sorted_vocab(d):
    """
    Sort the entire vocabulary (keys and keys of their value
    dictionaries).

    Input
    d -- dictionary mapping word-pairs to counts, created by
         cooccurrence_matrix(). We need only the keys for this step.

    Output
    vocab -- sorted list of strings
    """
    vocab = set([])
    for w1, val_dict in d.items():
        vocab.add(w1)
        for w2 in val_dict.keys():
            vocab.add(w2)
    vocab = sorted(list(vocab))
    return vocab

def get_vectors(d, vocab):
    """
    Interate through the vocabulary, creating the vector for each word
    in it.

    Input
    d -- dictionary mapping word-pairs to counts, created by
         cooccurrence_matrix()
    vocab -- sorted vocabulary created by get_sorted_vocab()

    Output
    vecs -- dictionary mapping words to their vectors.
    """    
    vecs = {}
    for w1 in vocab:
        v = []
        for w2 in vocab:
            wA, wB = sorted([w1, w2])
            v.append(d[wA][wB])
        vecs[w1] = array(v)
    return vecs

def cosim(v1, v2):
    """Cosine similarity between the two vectors v1 and v2."""
    num = dot(v1, v2)
    den = sqrt(dot(v1, v1)) * sqrt(dot(v2, v2))
    if den:
        return num/den
    else:
        return 0.0

def cosine_similarity_matrix(vocab, d):
    """
    Create the cosine similarity matrix.

    Input
    vocab -- a list of words derived from the keys of d
    d -- a two-dimensional defaultdict mapping word pairs to counts,
    as created by cooccurrence_matrix()

    Output
    cm -- a two-dimensional defaultdict mapping word pairs to their
    cosine similarity according to d    
    """
    cm = defaultdict(dict)
    vectors = get_vectors(d, vocab)
    for w1 in vocab:
        for w2 in vocab:
            cm[w1][w2] = cosim(vectors[w1], vectors[w2])
    return cm

def propagate(seedset, cm, vocab, a, iterations):
    """
    Propagates the initial seedset, with the cosine measures
    determining strength.
    
    Input
    seedset -- list of strings.
    cm -- cosine similarity matrix
    vocab -- the sorted vocabulary
    a -- the new value matrix
    iterations -- the number of iteration to perform

    Output
    pol -- dictionary mapping words to un-corrected polarity scores
    a -- the updated matrix
    """      
    for w_i in seedset:
        f = {}
        f[w_i] = True
        for t in range(iterations):
            for w_k in cm.keys():
                if w_k in f:
                    for w_j, val in cm[w_k].items():
                        # New value is max{ old-value, cos(k, j) } --- so strength
                        # can come from somewhere other th
                        a[w_i][w_j] = max([a[w_i][w_j], a[w_i][w_k] * cm[w_k][w_j]])
                        f[w_j] = True
    # Score tally.
    pol = {}
    for w in vocab:
        pol[w] = sum(a[w_i][w] for a_i in seedset)
    return [pol, a]

def graph_propagation(cm, vocab, positive, negative, iterations):
    """
    The propagation algorithm employing the cosine values.

    Input
    cm -- cosine similarity matrix (2-d dictionary) created by cosine_similarity_matrix()
    vocab -- vocabulary for cm
    positive -- list of strings
    negative -- list of strings
    iterations -- the number of iterations to perform

    Output:
    pol -- a dictionary form vocab to floats
    """
    pol = {}    
    # Initialize a.
    a = defaultdict(lambda : defaultdict(int))
    for w1, val_dict in cm.items():
        for w2 in val_dict.keys():
            if w1 == w2:
                a[w1][w2] = 1.0                    
    # Propagation.
    pol_positive, a = propagate(positive, cm, vocab, a, iterations)
    pol_negative, a = propagate(negative, cm, vocab, a, iterations)
    beta = sum(pol_positive.values()) / sum(pol_negative.values())
    for w in vocab:
        pol[w] = pol_positive[w] - (beta * pol_negative[w])
    return pol


def format_matrix(vocab, m):
    """
    For display purposes: builds an aligned and neatly rounded version
    of the two-dimensional dictionary m, assuming ordered values
    vocab. Returns string s.
    """
    s = ""
    sep = ""
    col_width = 15
    s += " ".rjust(col_width) + sep.join(map((lambda x : x.rjust(col_width)), vocab)) + "\n"
    for w1 in vocab:
        row = [w1]
        row += [round(m[w1][w2], 2) for w2 in vocab]
        s += sep.join(map((lambda x : str(x).rjust(col_width)), row)) + "\n"
    return s


    d = defaultdict(lambda : defaultdict(int))
    for text in my_text:
        #print(text)
        for i in text.split():  
            #print(i)
            for j in range(len(text)-1):
                
                #print(j)
                print(text[j])
                w1, w2 = sorted([text[i], text[j]])                
                d[w1][w2] += 1
    return d




d=cooccurrence_matrix(tuple_tuples)
vocab = get_sorted_vocab(d)
# Build the cosine matrix
cm = cosine_similarity_matrix(vocab, d)
# Sentiment propagation with simple seed sets.
prop = graph_propagation(cm, vocab, ["superb"], ["terrible"], 2)
# Display
print("Corpus:\n")
for text in corpus:
    print(" ".join(text))










## MODELS

#import for models
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import gutenberg
from operator import itemgetter
from copy import deepcopy
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt
from __future__ import division
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
import pandas as pd
import numpy as np
import re

df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")

# The code bellow was used to understand the Bag of Words, but during the report we decided to
# use another way to do this part, because it fits better with SVM

X_train = df_train_total["reviewText"]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_train_features = X_train_vectorized.todense()
X_train_vocab = vectorizer.vocabulary_

# Count words to see the most common words in the dataset
def words(text): return re.findall(r'\w+', text.lower())

WORDS_train = []
for text in df_train_total.get('reviewText'):
    WORDS_train.extend(words(text))

count_words = Counter(WORDS_train)
#check the top 50 words used in all reviews
cenas = Counter(WORDS_train).most_common(20)
#number of unique words in all reviews
len(Counter(WORDS_train))

count_vectorizer = CountVectorizer(stop_words='english')
#words, word_values = get_top_n_words(n_top_words=20, count_vectorizer=count_vectorizer, text_data=reindexed_data)
fig, ax = plt.subplots(figsize=(16,8))
ax.bar(range(len(words)), cenas)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words)
ax.set_title('Top Words')

#PLOT TOP 20 WORDS IN TRAINING SET
fig, ax = plt.subplots(figsize=(10,5))
ax.barh( range(len(cenas)), [t[1] for t in cenas] , height = 0.5 , align="center", color='#B2C5D8')
ax.set_yticks(range(len(cenas)))
ax.set_yticklabels(t[0] for t in cenas)
plt.title("Top 20 words in training set")
plt.tight_layout()
plt.show()

# BAG OF WORDS WITH NOUNS
wnl = WordNetLemmatizer()

def pos_tag_text(text):
    # convert Penn treebank tag to wordnet tag

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    text = word_tokenize(text)
    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

#SIMPLE TEST
pos_tag_text('distribute')


nouns = list()
nouns_all = list()
for text in df_train_total.get('reviewText'):
    pos_tagged_text = pos_tag_text(text)
    nouns.clear()
    for ( word, pos) in pos_tagged_text:
        if pos == 'N' or pos == 'n':
            nouns.append(word)
    nouns_all.append(' '.join(nouns))


df_nouns = df_train_total
df_nouns['reviewText'] = nouns_all

# SAVE THE BAG OF WORDS WITH NOUNS
df_nouns.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df nouns.pkl")
df_nouns = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df nouns.pkl")

# VERIFYING THE BIGRAMS
df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")

# FIRST ALTERNATIVE FOR BIGRAMS
df_aux = []
def words(text): return re.findall(r'\w+', text.lower())

for text in df_train_total.get('reviewText'):
    df_aux.extend(words(text))

# A SECOND ALTERNATIVE THAT PUT THE BIGRAMS INTO A DATA FRAME
bigrams= list()
bigrams_all = list()
for text in df_train_total.get('reviewText'):
    bigrams.clear()
    for word in text.split():
        bigrams.append(word)

    bigrams = list(ngrams(bigrams, 2))
    bigrams_all.append(tuple(bigrams))

#antes
bigrams = ngrams(df_aux, 2)
BigramFreq = Counter(bigrams_all)
get_bigrams_to_list = list(BigramFreq)

#tentative altrnativa de lista
get_bigrams_to_list = list(bigrams_all)
df_bigrams = df_train_total
df_bigrams['reviewText'] = get_bigrams_to_list

# MOST FREQUENT BIGRAMS
get_bigrams = BigramFreq.most_common(10)

# CREATE A PICKLE FILE TO STORE THE BIGRAMS ON THE DATASET
df_bigrams.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df bigrams.pkl")
df_bigrams = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df bigrams.pkl")

for text in df_bigrams.get('reviewText'):
    #for l in text:
    tuples_list = list(text)

# PLOT THE MOST FREQUENT BIGRAMS
fig, ax = plt.subplots(figsize=(10,5))
ax.barh( range(len(get_bigrams)), [t[1] for t in get_bigrams] , height = 0.5 , align="center", color='#B2C5D8')
ax.set_yticks(range(len(get_bigrams)))
ax.set_yticklabels([t[0] for t in get_bigrams])
plt.tight_layout()
plt.show()

# TF-IDF
tokenize = lambda doc: doc.lower().split(" ")

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

#in Scikit-Learn
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
print(sklearn_representation) # RESULT (document, word) tfidf value

## SVD

lsa =TruncatedSVD(n_components=100)
#bow normal and nouns
lsa.fit(X)
reviews_concepts_matrix =lsa.fit_transform(X)

#bigrams
lsa.fit(bow_train_features)
reviews_concepts_matrix =lsa.fit_transform(bow_train_features)

comps = lsa.components_

#Gives each concept with 10 words associated
terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse = True) [:10]
    print("Concept %d: " % i)
    for term in sortedTerms:
        print(term[0])
    print(" ")

# PLOT ELBOW
explained_variance = lsa.explained_variance_ratio_
print(lsa.explained_variance_ratio_)

singular_values = lsa.singular_values_

explained_variance_ratio_plot = np.cumsum(explained_variance)
plt.plot(explained_variance_ratio_plot )

plt.plot(singular_values)
plt.ylabel("Singular Values")
plt.xlabel("Concepts")

comps_reduzida = comps[:10][:]

# CATEGORIES LIST
categories_target = ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]
categories_target_nr = [0,1,2,3]
# FOR BOW - concepts list
concepts_list = ["Concept 1", "Concept 2", "Concept 3", "Concept 4", "Concept 5", "Concept 6", "Concept 7", "Concept 8"]
# FOR BOW NOUNS - concepts list
concepts_list = ["Concept 1", "Concept 2", "Concept 3", "Concept 4", "Concept 5", "Concept 6", "Concept 7", "Concept 8", "Concept 9", "Concept 10"]

df_reviews_concepts=pd.DataFrame(reviews_concepts_matrix)

# FOR BOW
new_list_labels = [word for word in df_train_total.get('Label')]
# FOR BOW NOUNS
new_list_labels = [word for word in df_nouns.get('Label')]

df_reviews_concepts['Label'] =  new_list_labels

df_teste =(df_reviews_concepts.groupby(['Label']).mean())

df_categories_concepts = df_teste.reset_index()
# for BOW normal
new_df =df_teste.drop(df_teste.columns[8:100], axis=1)
# for NOUNS
new_df =df_teste.drop(df_teste.columns[10:100], axis=1)

# PLOT THE GRID OF WEIGHTS OF EACH CONCEPT TO EACH CATEGORY
fig, ax = plt.subplots()
im = ax.imshow(new_df)

# We want to show all ticks...
ax.set_xticks(np.arange(len(concepts_list)))
ax.set_yticks(np.arange(len(categories_target)))
# ... and label them with the respective list entries
ax.set_xticklabels(concepts_list)
ax.set_yticklabels(categories_target)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

new_df_aux = new_df.reset_index()
new_df_aux =new_df_aux.drop(columns = ['Label'])


for i in range(len(categories_target)):
    for j in range(len(concepts_list)):
        new_df_aux.iat[i,j] ="%.3f" % new_df_aux.iat[i,j]
        text = ax.text(j, i, str(new_df_aux.iat[i,j]), ha="center", va="center", color="w")

ax.set_title("Catgories vs Concepts - Bag of words of nouns")
plt.figure(figsize=(10,5))
fig.tight_layout()
plt.show()

## Bag Of Words treated to Support Vector Machines

# IMPORTS
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools

df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")
df_test_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - test_clean_lemmatize.pkl")
#TRAIN CORPUS - for bigrams and bow normal
train_corpus = df_train_total['reviewText'].tolist()
train_label = df_train_total['Label'].tolist()
#TEST CORPUS - for the three models
test_corpus = df_test_total['reviewText'].tolist()
test_label = df_test_total['Label'].tolist()

#nouns corpus train
df_nouns = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df nouns.pkl")

train_corpus_nouns = df_nouns['reviewText'].tolist()
train_label_nouns = df_nouns['Label'].tolist()

#STEP 1 - CHOOSE ONE OF THE MODELS TO RUN
#-----------------------------------------------BOW normal
def bow_extractor(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()

vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW bigrams
def bow_extractor_bigrams(corpus, ngram_range=(2,2)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor_bigrams(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW Nouns
def bow_extractor_nouns(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor_nouns(train_corpus_nouns)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
vocab = bow_vectorizer.get_feature_names()

#--------------------------------------------------------------
#STEP 2 - EQUAL FOR ALL THREE MODELS

def get_metrics(true_labels, predicted_labels):

    print ('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels,
                                               predicted_labels),
                        2))
    print ('Precision:', np.round(
                        metrics.precision_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('Recall:', np.round(
                        metrics.recall_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('F1 Score:', np.round(
                        metrics.f1_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

mnb = MultinomialNB()
svm = LinearSVC(penalty='l2', C=1.0) # THE PARAMETERS FROM HANDOUT

# Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label,
                                           test_features=bow_test_features,
                                           test_labels=test_label)

cm = metrics.confusion_matrix(test_label, svm_bow_predictions)
pd.DataFrame(cm, index=range(0,4), columns=range(0,4))

#----------------------------------------------- PLOT
X_scaled = preprocessing.scale(cm)

df_cm = pd.DataFrame(cm, index = [i for i in ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]],
                  columns = [i for i in ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, cmap ="Blues")
sns.palplot(sns.color_palette("GnBu_d"))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm[i,j] ="%.3f" % cm.iat[i,j]
        plt.text(j, i, cm[i, j], horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
#--------------END OF PLOT FUNCTION

categories_target = ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]

class_names = categories_target
 
plt.figure(figsize = (10,7))
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

