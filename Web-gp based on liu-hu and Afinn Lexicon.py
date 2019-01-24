
# coding: utf-8
import math
import re
import sys
import pandas as pd
import numpy as np

# In[16]:
df_train = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - clean_lemmatize_with_stop_w_label.pkl")
df_test = pd.read_pickle("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\Pickle\\4 - clean_lemmatize_with_stop.pkl")


# In[25]:

# WEB_GP LEXICON 10370 WORDS, BALANCED ACCORDING TO LIU AND HU LEXICON (1 and -1)
sentiment_dict = {}
for line in open("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT2\\lexicon4words_final.txt", encoding="utf-8"):
    word, score = line.split(',')
    sentiment_dict[word] = int(score)


# In[26]:
sentiment_dict


# In[27]:

## Test WEB_GP Lexicon 
pattern_split = re.compile(r"\W+")

def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence. 
    """
    final_sentiment = []
    words = pattern_split.split(text.lower())
    pos_words = 0
    neg_words = 0
    #sentiments = [sentiment_dict.get(word, 0) for word in words]
    for sent in words:
        if sent in sentiment_dict.keys():
            if sentiment_dict[sent] >= 1:
                pos_words += 1
            elif sent in sentiment_dict:
                neg_words += 1
            
    if pos_words > neg_words:
        final_sentiment.append('Positive')
        #print('Positive')
    elif pos_words < neg_words:
        final_sentiment.append('Negative')
        #print('Negative')
    elif pos_words == neg_words:
        final_sentiment.append('Neutral')
        #print('Neutral')
        # How should you weight the individual word sentiments? 
        # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
        #sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))

    #else:
        #sentiment = 0
    return final_sentiment


# In[61]:
df_total = pd.concat([df_train, df_test])

# In[65]:
list_df = []
for item in df_train['reviewText']:
    list_df.append(item)

labels = []
for review in list_df:
    labels.append(sentiment(review))
    
labels = [item for sublist in labels for item in sublist]


# In[62]:
# Test Afinn
from afinn import Afinn
afn = Afinn(emoticons=True) 


labels_affin = []
for review in list_df:
    if afn.score(review) >= 1:
        labels_affin.append("Positive")
    elif afn.score(review) < 0:
        labels_affin.append("Negative")
    else:
        labels_affin.append("Neutral")
#    labels.append(afn.score(review))
