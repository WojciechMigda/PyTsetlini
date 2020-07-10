# -*- coding: utf-8 -*-

# based on https://github.com/cair/pyTsetlinMachine/blob/dfaa7f36a5fa5cc852645277605358ae4d955898/examples/IMDbTextCategorizationDemo.py

from scipy import sparse
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES=5000

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
    terms = []
    for word_id in train_x[i]:
        terms.append(id_to_word[word_id])

    for N in range(1,MAX_NGRAM+1):
        grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
        for gram in grams:
            phrase = " ".join(gram)

            if phrase in vocabulary:
                vocabulary[phrase] += 1
            else:
                vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
    if vocabulary[phrase] < 10:
        continue

    phrase_bit_nr[phrase] = bit_nr
    bit_nr_phrase[bit_nr] = phrase
    bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint8)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint8)
for i in range(train_y.shape[0]):
    terms = []
    for word_id in train_x[i]:
        terms.append(id_to_word[word_id])

    for N in range(1,MAX_NGRAM+1):
        grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
        for gram in grams:
            phrase = " ".join(gram)
            if phrase in phrase_bit_nr:
                X_train[i,phrase_bit_nr[phrase]] = 1

    Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint8)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint8)

for i in range(test_y.shape[0]):
    terms = []
    for word_id in test_x[i]:
        terms.append(id_to_word[word_id])

    for N in range(1,MAX_NGRAM+1):
        grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
        for gram in grams:
            phrase = " ".join(gram)
            if phrase in phrase_bit_nr:
                X_test[i,phrase_bit_nr[phrase]] = 1

    Y_test[i] = test_y[i]

X_train = sparse.csr_matrix(X_train)
X_test = sparse.csr_matrix(X_test)

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

output_test = np.c_[X_test.toarray(), Y_test]
np.savetxt("IMDBTestData.txt.gz", output_test, fmt="%d")

output_train = np.c_[X_train.toarray(), Y_train]
np.savetxt("IMDBTrainingData.txt.gz", output_train, fmt="%d")
