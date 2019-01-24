# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:21:21 2018

@author: Nicole Rita
"""
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

################ KBEST BASED ON BAG OF WORDS #####################################
df_kbest = pd.concat([df_test_total_2, df_train_total_2])
corpus_kbest = df_kbest['reviewText'].tolist()
label_kbest = df_kbest['overall'].tolist()

vectorizer = CountVectorizer(lowercase=True, analyzer = "word")
full_matrix = vectorizer.fit_transform(corpus_kbest)

selector = SelectKBest(chi2, k=2000)
selector.fit(full_matrix, label_kbest)
top_words = selector.get_support().nonzero()

print(full_matrix.shape)

chi_matrix = full_matrix[:,top_words[0]]

features = np.hstack([chi_matrix.todense()])

train_rows = 280000
# Set a seed to get the same "random" shuffle every time.
random.seed(1)

# Shuffle the indices for the matrix.
indices = list(range(features.shape[0]))
random.shuffle(indices)

# Create train and test sets.
train = features[indices[:train_rows], :]
test = features[indices[train_rows:], :]


train_label = df_kbest['overall'].iloc[indices[:train_rows]]
test_label = df_kbest['overall'].iloc[indices[train_rows:]]
train = np.nan_to_num(train)

svm_bow_kbest_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=train,
                                           train_labels=train_label,
                                           test_features=test,
                                           test_labels=test_label)








