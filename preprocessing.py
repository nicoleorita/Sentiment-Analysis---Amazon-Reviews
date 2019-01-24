# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:34:27 2018

@author: Nicole Rita
https://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost
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
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

#-------------------------------2nd project

df_train_total_2 = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - clean_lemmatize.pkl")
df_test_total_2 = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - test_clean_lemmatize.pkl")

'''
my_list = [('this', 'p'), ('product', 'n'), ('is', 'v'), ('funny', 'a'), ('and', 'p'),('scary', 'a'), ('and', 'p'), ('it', 'p'), ('is', 'v'), ('an', 'p'), ('insanely', 'a'),('great', 'a'), ('movie', 'n'), ('loved', 'v'), ('the', 'p'),('exciting', 'a'), ('ending', 'n')]

my_text = ["I am a product of bags and drops beautiful empty legendary epic makeup cute watch cookies", "baby went to buy me a beautiful set of epic makeup", "legendary so epic", "legendary epic"]
my_something = [['hey', 'hello', 'i_am', 'no_dont'], ['limao', 'limon_cenas']]
'''
#-----------------------------------------------DF ADJECTIVE NOUNS----------------------
list_aux = list()
lista_end = list()
lista_final = list()

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

#-------------------------------------------DF NOUNS, ADJECTIVES, ADVERBS----------------------------------------
from nltk import pos_tag

lista_naa = list()
for l in df_test_total_2.get('reviewText'):
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

dftst_adj_nouns_adv = df_test_total_2
dftst_adj_nouns_adv['reviewText'] = lista_naa

join_reviews= [ ' '.join(l) for l in dftst_adj_nouns_adv['reviewText']]
dftst_adj_nouns_adv['reviewText'] = join_reviews


# SAVE THE DATAFRAME WITH ADJECTIVES, NOUNS & ADVERBS
dftst_adj_nouns_adv.to_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\6 - dftst_adjective_nouns_adv.pkl")
df_adj_nouns_adv = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\6 - df_adjective_nouns_adv.pkl")



##############################################################################################################
                                                  # BOW #
##############################################################################################################

# The code bellow was used to understand the Bag of Words, but during the report we decided to
# use another way to do this part, because it fits better with SVM

X_train = df_train_total_2["reviewText"]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
X_train_vectorized = vectorizer.fit_transform(train_corpus_adj_nouns_adv)
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

##############################################################################################################
                                                  # VISUAL TFIDF #
##############################################################################################################

from __future__ import division
import string
import math
 
all_documents = df_train_total_2['reviewText'].tolist()

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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools



#TRAIN CORPUS - for bow, TFIDF normal & K-BEST
train_corpus = df_train_total_2['reviewText'].tolist()
train_label = df_train_total_2['overall'].tolist()

     
train_label = [int(i) for i in train_label]
test_label = [int(i) for i in test_label]

#TEST CORPUS - for the three models
test_corpus = df_test_total_2['reviewText'].tolist()
test_label = df_test_total_2['overall'].tolist()

#adjectives and nouns corpus train
train_corpus_adj_nouns = df_adj_nouns['reviewText'].tolist()
train_label_adj_nouns = df_adj_nouns['overall'].tolist()
train_label_adj_nouns = [int(i) for i in train_label_adj_nouns]

#adjectives, adverbs and nouns corpus train
train_corpus_adj_nouns_adv = df_adj_nouns_adv['reviewText'].tolist()
train_label_adj_nouns_adv = df_adj_nouns_adv['overall'].tolist()

test_corpus_adj_nouns_adv = dftst_adj_nouns_adv['reviewText'].tolist()
test_label_adj_nouns_adv = dftst_adj_nouns_adv['overall'].tolist()

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
kbest_features = kbest_classifier.fit_transform(tfidf_matrixx, test_label)


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_transformer(bow_matrix):
    
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix

def tfidf_extractor(corpus, ngram_range=(1,1)):
    
    vectorizer = TfidfVectorizer(max_features = 5000, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)
tfidf_matrixx = tfidf_test_features.todense()


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

# Support Vector Machine with bag of adj noun adv
svm_bow_adj_noun_adv_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label_adj_nouns,
                                           test_features=bow_test_features,
                                           test_labels=test_label)

svm_bow_adj_noun_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label_adj_nouns,
                                           test_features=bow_test_features,
                                           test_labels=test_label)

svm_bow_kbest_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label,
                                           test_features=kbest_features,
                                           test_labels=test_label)

# SVM with K-Best tfidf
svm_bow_kbest_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_train_features,
                                           train_labels=train_label,
                                           test_features=X_new,
                                           test_labels=test_label)

# Support Vector Machine with tfidf features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_train_features,
                                           train_labels=train_label,
                                           test_features=tfidf_test_features,
                                           test_labels=test_label)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

cm = metrics.confusion_matrix(test_label, svm_bow_predictions)
pd.DataFrame(cm, index=range(0,20), columns=range(0,20))

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, svm_sentiment_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = ["positive", "negative", "neutral"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

