import cPickle
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta
from keras.constraints import unitnorm, maxnorm
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence

from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

from sklearn.metrics import roc_auc_score

# This import to resolve some errors with tf version on office server
import tensorflow as tf
tf.python.control_flow_ops = tf

def get_idx_from_sent(sent, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    # pad = kernel_size - 1
    # for i in xrange(pad):
    #     x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    # print type(x), x
    # x = sequence.pad_sequences(x, maxlen=max_l)
    return x

def make_idx_data(revs, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, max_l, kernel_size)
        sent.append(rev['y'])
        if rev['split'] == 1:
            train.append(sent)
        elif rev['split'] == 0:
            val.append(sent)
        else:
            test.append(sent)
    train = np.array(train, dtype=np.int)
    val = np.array(val, dtype=np.int)
    test = np.array(test, dtype=np.int)
    return [train, val, test]


print "loading data..."
x = cPickle.load(open("bigger_health-train-val-test2.pickle", "rb"))
revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
print "data loaded!"


datasets = make_idx_data(revs, word_idx_map, max_l=247, kernel_size=5)

# Train data preparatio
N = datasets[0].shape[0]
conv_input_width = W.shape[1]
conv_input_height = int(datasets[0].shape[1]-1)

# For each word write a word index (not vector) to X tensor
train_X = np.zeros((N, conv_input_height), dtype=np.int)
train_Y = np.zeros((N, 2), dtype=np.int)
for i in xrange(N):
    for j in xrange(conv_input_height):
        train_X[i, j] = datasets[0][i, j]
    train_Y[i, datasets[0][i, -1]] = 1
    
    
print 'train_X.shape = {}'.format(train_X.shape)
print 'train_Y.shape = {}'.format(train_Y.shape)

Nv = datasets[1].shape[0]

val_X = np.zeros((Nv, conv_input_height), dtype=np.int)
val_Y = np.zeros((Nv, 2), dtype=np.int)
for i in xrange(Nv):
    for j in xrange(conv_input_height):
        val_X[i, j] = datasets[1][i, j]
    val_Y[i, datasets[1][i, -1]] = 1
    
print 'val_X.shape = {}'.format(val_X.shape)
print 'val_Y.shape = {}'.format(val_Y.shape)



# N_fm = 300
# kernel_size = 8
model = Sequential()
model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
model.add(LSTM(300))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


#optimizer is adadelta and loss function is categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

#
# epoch = 0
# val_acc = []
# val_auc = []

# N_epoch = 3

# for i in xrange(N_epoch):
#     model.fit(train_X, train_Y, batch_size=50, nb_epoch=1, verbose=1, show_accuracy=True)
#     output = model.predict_proba(val_X, batch_size=10, verbose=1)
#     vacc = np.max([np.sum((output[:,1]>t)==(val_Y[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
#     vauc = roc_auc_score(val_Y, output[:,0])
#     val_acc.append(vacc)
#     val_auc.append(vauc)
#     print 'Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc)
#     epoch += 1
    
# print '{} epochs passed'.format(epoch)
# print 'Accuracy on validation dataset:'
# print val_acc
# print 'AUC on validation dataset:'
# print val_auc

# model.save_weights('cnn_3epochs2.model')

model.fit(train_X, train_Y, batch_size=50, nb_epoch=10, validation_data=(val_X, val_Y))
model.save_weights('cnn_3epochs2.model')


# Test data preparation
Nt = datasets[2].shape[0]

test_X = np.zeros((Nt, conv_input_height), dtype=np.int)
test_Y = np.zeros((Nt, 2), dtype=np.int)
for i in xrange(Nt):
    for j in xrange(conv_input_height):
        test_X[i, j] = datasets[2][i, j]
    test_Y[i, datasets[2][i, -1]] = 1    
        
    
print 'test_X.shape = {}'.format(test_X.shape)
print 'test_Y.shape = {}'.format(test_Y.shape)

score, acc = model.evaluate(test_X, test_Y,
                            batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)
