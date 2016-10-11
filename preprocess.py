import random
import csv
import os
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from nltk import word_tokenize
import nltk
import itertools
import operator

def prepare_data(reviews, vocabulary_size=2000):
    # SENTENCE_START_TOKEN = "SENTENCE_START"
    # SENTENCE_END_TOKEN = "SENTENCE_END"
    UNKNOWN_TOKEN = "UNKNOWN_TOKEN"    

    sentences = []
    tokens = []
    tokenized_sentences = []

    for x in reviews:
        tokens= word_tokenize(x.decode("utf-8").lower())
        # Can do some better cleaning here like change n't to not, 've,to have etc. 

        tokenized_sentences.append(tokens)
        sentences.append(' '.join(tokens).strip())
        

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size]
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))


    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    # Correct the logic here: y_train should be binary classifier and ranks out of 5
    X_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])

    ## Create ndarrays (binary + ratings) and save it to a file/
    ## Check for that file at the start of this method..to prevent useless processing.
    
    return X_train, word_to_index, index_to_word

    # return sentences

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def changeToBinary(i):
    return 0 if i <= 3 else 1 

text=[]
helpful=[]
helpful_bin = []
knowledge=[]
knowledge_bin = []
staff_bin = []
staff=[]

input_data = []

files = [os.path.join("data/", f) for f in os.listdir('data/') if os.path.isfile(os.path.join("data/", f))]

for f in files[:20]:
    if f.endswith(".txt"):
        with open(f) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                if(len(line) > 6 and RepresentsInt(line[3]) and RepresentsInt(line[5]) and RepresentsInt(line[6]) ):
                    text.append(line[2])
                    staff.append(int(line[3]))
                    helpful.append(int(line[5]))
                    knowledge.append(int(line[6]))
                    
                    staff_bin.append(changeToBinary(int(line[3])))
                    helpful_bin.append(changeToBinary(int(line[5])))
                    knowledge_bin.append(changeToBinary(int(line[6])))

                else: 
                    pass
                    # write reject lines to a file for analyzing later

print "Rankings are numbers from 1 to 5:", set(knowledge) == set(staff) == set (helpful) == set([1,2,3,4,5])
print "Binary classifiers are either 0 or 1:", set(knowledge_bin) == set(staff_bin) == set (helpful_bin) == set([0,1])

X, word_to_index, index_to_word = prepare_data(text)

splitSize = 20 

l = len(X)

X_test, X_train = X[:l/splitSize], X[l/splitSize:]

staff_test, staff_train = staff[:(l/splitSize)], staff[(l/splitSize):]
staff_bin_test, staff_bin_train = staff_bin[:(l/splitSize)], staff_bin[(l/splitSize):]

knowledge_test, knowledge_train = knowledge[:(l/splitSize)], knowledge[(l/splitSize):]
knowledge_bin_test, knowledge_bin_train = knowledge_bin[:(l/splitSize)], knowledge_bin[(l/splitSize):]

helpful_test, helpful_train = helpful[:(l/splitSize)], helpful[(l/splitSize):]
helpful_bin_test, helpful_bin_train = helpful_bin[:(l/splitSize)], helpful_bin[(l/splitSize):]

print "Comparing mean of test vs train for 5-scale ratings - staff:", sum(staff_test)/ float(len(staff_test)), sum(staff_train)/ float(len(staff_train))
print "Comparing mean of test vs train for 5-scale ratings - helpful:", sum(helpful_test)/ float(len(helpful_test)), sum(helpful_train)/ float(len(helpful_train))
print "Comparing mean of test vs train for 5-scale ratings - knowledge:", sum(knowledge_test)/ float(len(knowledge_test)), sum(knowledge_train)/ float(len(knowledge_train))