# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:02:15 2018

@author: Nicole Rita
"""

import sys
from collections import defaultdict
from operator import itemgetter
from numpy import dot, sqrt, array
import numba
from numba import jit
import pandas as pd
import time
import random

def timer(f, *args, **kwargs):
    start = time.clock()
    ans = f(*args, **kwargs)
    return ans, time.clock() - start
def report(fs, *args, **kwargs):
    ans, t = timer(fs[0], *args, **kwargs)
    print('%s: %.1f' % (fs[0].__name__, 1.0))
    for f in fs[1:]:
        ans_, t_ = timer(f, *args, **kwargs)
        print('%s: %.1f' % (f.__name__, t/t_))
        
        
################################# CREATE THE DATA TO USE IN WEB_GP
df_train_total = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - clean_lemmatize_with_stop_w_label.pkl")


train_json = [ 
    dict([
        (colname, row[i]) 
        for i,colname in enumerate(df_train_total.columns)
    ])
    for row in df_train_total.values
]

overall1 = list()
overall2 = list()
overall3 = list()
overall4 = list()
overall5 = list()

for j in train_json:
       if j.get('overall') == 1.0 and j.get('Label') == 'Beauty':
              overall1.append(j)
       elif j.get('overall') == 2.0 and j.get('Label') == 'Beauty': 
            overall2.append(j)
       elif j.get('overall') == 3.0 and j.get('Label') == 'Beauty':
            overall3.append(j)
       elif j.get('overall') == 4.0 and j.get('Label') == 'Beauty':
            overall4.append(j)
       elif j.get('overall') == 5.0 and j.get('Label') == 'Beauty':
            overall5.append(j)

def calculate_percentage ():    
    num_to_select1 = 1400  
    num_to_select2 = 1400
    num_to_select3 = 1400
    num_to_select4 = 1400
    num_to_select5 = 1400
    
    final_list = list()
    final_list.extend(random.sample(overall1, num_to_select1))
    final_list.extend(random.sample(overall2, num_to_select2))
    final_list.extend(random.sample(overall3, num_to_select3))
    final_list.extend(random.sample(overall4, num_to_select4))
    final_list.extend(random.sample(overall5, num_to_select5))
    return final_list

df_1percent = pd.DataFrame(calculate_percentage())
df_grocery_1percent = pd.DataFrame(calculate_percentage())
df_movies_tv_1percent = pd.DataFrame(calculate_percentage())
df_toys_games_1percent = pd.DataFrame(calculate_percentage())

df_train_web_gp = pd.concat([df_1percent, df_grocery_1percent, df_movies_tv_1percent, df_toys_games_1percent])

df_train_web_gp.to_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\df_train_web_gp.pkl")
df_train_web_gp = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\df_train_web_gp.pkl")

################# WEB_GP

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
@jit
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
@jit
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
@jit
def cosim(v1, v2):
    """Cosine similarity between the two vectors v1 and v2."""
    num = dot(v1, v2)
    den = sqrt(dot(v1, v1)) * sqrt(dot(v2, v2))
    if den:
        return num/den
    else:
        return 0.0
 

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

# Create a list of lists and tranform into tuple of tuples
web_gp_list = []
for text in df_train_web_gp.get('reviewText'):
    txt = text.split()
    web_gp_list.append(txt)
    
tuple_tuples = tuple(tuple(txt) for txt in web_gp_list)
# Build the co-occurrence matrix.
d = cooccurrence_matrix(tuple_tuples)
%timeit cooccurrence_matrix(tuple_tuples)
# Get the vocab
vocab = get_sorted_vocab(d)
%timeit get_sorted_vocab(d)
# Build the cosine matrix
cm = cosine_similarity_matrix(vocab, d)
%timeit cosine_similarity_matrix(vocab, d)
# Sentiment propagation with simple seed sets.
prop = graph_propagation(cm, vocab, ["good", "nice", "beautifull", "good"], ["horrible", "disappointed", "bad", "terrible"], 2)

# Display.
print("Corpus:\n")
for text in tuple_tuples:
    print(" ".join(text))

print("Co-occurence matrix:\n")
print(format_matrix(vocab, d))

print("Cosine similarity matrix:\n")
print(format_matrix(vocab, cm))


print("Propagated polarity: {superb} and {terrible} as seeds, 2 iterations\n")
for key, val in sorted(prop.items(), key=itemgetter(1), reverse=True):
    print(key, val)
    
##################### USING LIU AND HU LEXICON
import nltk

def _show_plot(x_values, y_values, x_labels=None, y_labels=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('The plot function requires matplotlib to be installed.'
                         'See http://matplotlib.org/')

    plt.locator_params(axis='y', nbins=3)
    axes = plt.axes()
    axes.yaxis.grid()
    plt.plot(x_values, y_values, 'ro', color='red')
    plt.ylim(ymin=-1.2, ymax=1.2)
    plt.tight_layout(pad=5)
    if x_labels:
        plt.xticks(x_values, x_labels, rotation='vertical')
    if y_labels:
        plt.yticks([-1, 0, 1], y_labels, rotation='horizontal')
    # Pad margins so that markers are not clipped by the axes
    plt.margins(0.2)
    plt.show()
    
def demo_liu_hu_lexicon(sentence, plot=False):
    """
    Basic example of sentiment classification using Liu and Hu opinion lexicon.
    This function simply counts the number of positive, negative and neutral words
    in the sentence and classifies it depending on which polarity is more represented.
    Words that do not appear in the lexicon are considered as neutral.

    :param sentence: a sentence whose polarity has to be classified.
    :param plot: if True, plot a visual representation of the sentence polarity.
    """
    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank
    final_sentiment = []

    tokenizer = treebank.TreebankWordTokenizer()
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]

    x = list(range(len(tokenized_sent))) # x axis for the plot
    y = []

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
            y.append(1) # positive
        elif word in opinion_lexicon.negative():
            neg_words += 1
            y.append(-1) # negative
        else:
            y.append(0) # neutral

    if pos_words > neg_words:
        #final_sentiment.append('Positive')
        print('Positive')
    elif pos_words < neg_words:
        #final_sentiment.append('Negative')
        print('Negative')
    elif pos_words == neg_words:
        #final_sentiment.append('Neutral')
        print('Neutral')

    if plot == True:
        _show_plot(x, y, x_labels=tokenized_sent, y_labels=['Negative', 'Neutral', 'Positive'])

senteca = ['i have buy this before from another website and love the smell this particular one receive thru amazon have bad scent do not know what be up with that']
sentiment = []
for t in senteca:
       demo_liu_hu_lexicon(t, plot=True)


