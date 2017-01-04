import os
print os.getcwd()

import csv
import os
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from keras.preprocessing import sequence



def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False



text=[]
helpful=[]
helpful_categorical = []
knowledge=[]
staff=[]
knowledge_categorical=[]
staff_categorical=[]
ii=0
jj=0
files = [os.path.join("data/", f) for f in os.listdir('data/') if os.path.isfile(os.path.join("data/", f))]
for f in files:
    if f.endswith(".txt"):
        with open(f) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                ii=ii+1
                if(len(line) > 6 and RepresentsInt(line[3]) and RepresentsInt(line[5]) and RepresentsInt(line[6]) ):
                    jj=jj+1
#                     print (line[3], line[5], int(line[6]))
                    if int(line[3])>=3:
                        staff.append(int(1))
                    if int(line[3])<3:
                        staff.append(int(0))
                    if int(line[5])>=3:
                        helpful.append(int(1))
                    if int(line[5])<3:
                        helpful.append(int(0))                        
                    if int(line[6])>=3:
                        knowledge.append(int(1))
                    if int(line[6])<3:
                        knowledge.append(int(0))                                            
                    helpful_categorical.append(int(line[5])-1)
                    knowledge_categorical.append(int(line[6])-1)
                    staff_categorical.append(int(line[3])-1)
                    text.append(line[2])
        
print "Total lines:", ii ,"\nValid lines:", jj        
print "Text: ",len(text)

print "Helpful", len(helpful)
print "Helpful_cat", len(helpful_categorical)

print "Staff", len(staff)
print "Staff_cat", len(staff_categorical)

print "Knowledge", len(knowledge)
print "Knowledge_cat", len(knowledge_categorical)


text_train=text[:34000]
text_test=text[34000:]

helpful_train=helpful[:34000]
helpful_test=helpful[34000:]

helpful_cat_train=helpful_categorical[:34000]
helpful_cat_test=helpful_categorical[34000:]

staff_train=staff[:34000]
staff_test=staff[34000:]

staff_cat_train=staff_categorical[:34000]
staff_cat_test=staff_categorical[34000:]


knowledge_train=knowledge[:34000]
knowledge_test=knowledge[34000:]

knowledge_cat_train=knowledge_categorical[:34000]
knowledge_cat_test=knowledge_categorical[34000:]    




def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()



def build_data_train_test(train_ratio = 0.8, clean_string=True):
    """
    Loads data and split into train and test sets.
    """
    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set
    for i in xrange(len(text_train)):
        line = text_train[i]
        #helpful
        y_helpful = helpful_train[i]
        y_helpful_cat = np.zeros(5)
        y_helpful_cat[helpful_cat_train[i]] =1
        #staff
        y_staff = staff_train[i]
        y_staff_cat = np.zeros(5)
        y_staff_cat[staff_cat_train[i]] =1
        #knowledge
        y_knowledge = knowledge_train[i]
        y_knowledge_cat = np.zeros(5)
        y_knowledge_cat[knowledge_cat_train[i]] =1        
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(' '.join(rev))
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1

        datum  = {'y_helpful': y_helpful, 
                  'y_helpful_cat': y_helpful_cat,
                  'y_staff': y_staff, 
                  'y_staff_cat': y_staff_cat,
                  'y_knowledge': y_knowledge, 
                  'y_knowledge_cat': y_knowledge_cat,
                  'text': orig_rev,
                  'num_words': len(orig_rev.split()),
                  'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)
    # Pre-process test data set
    for i in xrange(len(text_test)):
        line = text_test[i]
        #helpful
        y_helpful = helpful_test[i]
        y_helpful_cat = np.zeros(5)
        y_helpful_cat[helpful_cat_test[i]] =1
        #staff
        y_staff = staff_test[i]
        y_staff_cat = np.zeros(5)
        y_staff_cat[staff_cat_test[i]] =1
        #knowledge
        y_knowledge = knowledge_test[i]
        y_knowledge_cat = np.zeros(5)
        y_knowledge_cat[knowledge_cat_test[i]] =1
        
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(' '.join(rev))
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {'y_helpful': y_helpful, 
                  'y_helpful_cat': y_helpful_cat,
                  'y_staff': y_staff, 
                  'y_staff_cat': y_staff_cat,
                  'y_knowledge': y_knowledge, 
                  'y_knowledge_cat': y_knowledge_cat,
                  'text': orig_rev,
                  'num_words': len(orig_rev.split()),
                  'split': -1}
        revs.append(datum)      
    return revs, vocab    



revs, vocab = build_data_train_test(train_ratio=0.8, clean_string=True)
max_l = np.max(pd.DataFrame(revs)['num_words'])
print 'data loaded!'
print 'number of sentences: ' + str(len(revs))
print 'vocab size: ' + str(len(vocab))
print 'max sentence length: ' + str(max_l)    


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)
    W[0] = np.zeros(k, dtype=np.float32)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        print header
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size): #Change this to full on a bigger machine
            #print line
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
            if (line%100000==0):
                print line
    return word_vecs



def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    i=0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            i=i+1
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)              
    print "Total words not in vec:", i


w2v_file = 'GoogleNews-vectors-negative300.bin'
# w2v = {} #load bin vec here later
w2v = load_bin_vec(w2v_file, vocab) 
add_unknown_words(w2v, vocab)
W, word_idx_map = get_W(w2v)
cPickle.dump([revs, W, word_idx_map, vocab], open('text-w-map-vocab.pickle', 'wb'))
print 'dataset created!'




def get_idx_from_sent(sent, word_idx_map, max_l=51):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x




def make_idx_data(revs, word_idx_map, max_l=51):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    
    y_helpful_train, y_helpful_val, y_helpful_test = [], [] , []
    y_helpful_cat_train, y_helpful_cat_val, y_helpful_cat_test = [], [] , []

    y_staff_train, y_staff_val, y_staff_test = [], [] , []
    y_staff_cat_train, y_staff_cat_val, y_staff_cat_test = [], [] , []

    y_knowledge_train, y_knowledge_val, y_knowledge_test = [], [] , []
    y_knowledge_cat_train, y_knowledge_cat_val, y_knowledge_cat_test = [], [] , []
    
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, max_l)
        if rev['split'] == 1:
            train.append(sent)
            #helpful
            y_helpful_train.append(int(rev["y_helpful"]))
            y_helpful_cat_train.append(rev["y_helpful_cat"])
            #staff
            y_staff_train.append(int(rev["y_staff"]))
            y_staff_cat_train.append(rev["y_staff_cat"])
            #knowledge
            y_knowledge_train.append(int(rev["y_knowledge"]))
            y_knowledge_cat_train.append(rev["y_knowledge_cat"])
        elif rev['split'] == 0:
            val.append(sent)
            #helpful
            y_helpful_val.append(int(rev["y_helpful"]))
            y_helpful_cat_val.append(rev["y_helpful_cat"])
            #staff
            y_staff_val.append(int(rev["y_staff"]))
            y_staff_cat_val.append(rev["y_staff_cat"])
            #knowledge
            y_knowledge_val.append(int(rev["y_knowledge"]))
            y_knowledge_cat_val.append(rev["y_knowledge_cat"])
        elif rev['split'] == -1:
            test.append(sent)
            #helpful
            y_helpful_test.append(int(rev["y_helpful"]))
            y_helpful_cat_test.append(rev["y_helpful_cat"])
            #staff
            y_staff_test.append(int(rev["y_staff"]))
            y_staff_cat_test.append(rev["y_staff_cat"])
            #knowledge
            y_knowledge_test.append(int(rev["y_knowledge"]))
            y_knowledge_cat_test.append(rev["y_knowledge_cat"])
    
    train = sequence.pad_sequences(train, maxlen=max_l)
    y_helpful_train = np.array(y_helpful_train, dtype=np.int)
    y_helpful_cat_train = np.array(y_helpful_cat_train, dtype=np.int)
    y_staff_train = np.array(y_staff_train, dtype=np.int)
    y_staff_cat_train = np.array(y_staff_cat_train, dtype=np.int)
    y_knowledge_train = np.array(y_knowledge_train, dtype=np.int)
    y_knowledge_cat_train = np.array(y_knowledge_cat_train, dtype=np.int)
      
    val = sequence.pad_sequences(val, maxlen=max_l)
    y_helpful_val = np.array(y_helpful_val, dtype=np.int)
    y_helpful_cat_val = np.array(y_helpful_cat_val, dtype=np.int)
    y_staff_val = np.array(y_staff_val, dtype=np.int)
    y_staff_cat_val = np.array(y_staff_cat_val, dtype=np.int)
    y_knowledge_val = np.array(y_knowledge_val, dtype=np.int)
    y_knowledge_cat_val = np.array(y_knowledge_cat_val, dtype=np.int)

    test = sequence.pad_sequences(test, maxlen=max_l)
    y_helpful_test = np.array(y_helpful_test, dtype=np.int)
    y_helpful_cat_test = np.array(y_helpful_cat_test, dtype=np.int)
    y_staff_test = np.array(y_staff_test, dtype=np.int)
    y_staff_cat_test = np.array(y_staff_cat_test, dtype=np.int)
    y_knowledge_test = np.array(y_knowledge_test, dtype=np.int)
    y_knowledge_cat_test = np.array(y_knowledge_cat_test, dtype=np.int)
    
    return [train, val, test, y_helpful_train, y_helpful_val, y_helpful_test, 
                              y_helpful_cat_train, y_helpful_cat_val, y_helpful_cat_test,
                              y_staff_train, y_staff_val, y_staff_test, 
                              y_staff_cat_train, y_staff_cat_val, y_staff_cat_test,
                              y_knowledge_train, y_knowledge_val, y_knowledge_test, 
                              y_knowledge_cat_train, y_knowledge_cat_val, y_knowledge_cat_test]




datasets = make_idx_data(revs, word_idx_map, max_l=247)

cPickle.dump(datasets, open('train-test-val.pickle', 'wb'))
print 'dataset created!'
