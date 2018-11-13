from __future__ import print_function
from parser import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adamax, Adam, Nadam
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import one_hot
import numpy as np
import random
import sys
import time
import os
import argparse
import json

with open('config.json', 'rb') as configfile:
    config = json.load(configfile)

#print 'using config', config_name
for k in sorted(config.keys()):
    print('', k, '->', config[k])

count = config['num_games']
game, orient, weight, iterations = [], [], [], []
for c in range(count):
    game.append(str(config['games'][c]))
    orient.append(str(config['orientation'][c]))
    weight.append(float(config['weights'][c]))
    iterations.append(int(config['iterations'][c]))

level_size = int(config['size'])
segment_size = int(config['segment_size'])
blend = config['blend']
weighted = bool(int(config['weighted']))
seed = game[int(config['seed'])]
print("Blend: ", blend, "\tWeighted: ", str(weighted), "\tIterations: ", iterations)

def make_training_data(folder,orientation,duplication):
    X, Y = [], []
    levels = []
    for file in os.listdir(folder):
        with open(os.path.join(folder,file),'rb') as infile:
            level = []
            for line in infile:
                level.append(list(str(line.rstrip(),'utf-8')))
                #level.append(str(line.rstrip(),'utf-8'))
            levels.append(level)
    
    new_levels = []
    for ii in range(duplication):
        new_levels += levels
    levels = new_levels
    #random.shuffle(levels)
    

    outstr = ''
    print(len(levels))
    for level in levels:
        level_str = []
        width = len(level[0])
        height = len(level)
        if orientation == 'C':
            for column in range(width):
                line_str = ''
                for row in range(height):
                    line_str += str(level[row][column])
                X.append(line_str)
                Y.append(0)
        else:
            for row in range(height):
                line_str = ''
                for column in range(width):
                    line_str += str(level[row][column])
                X.append(line_str)
                Y.append(1)
        #print(level_str)
    return X, Y



X_SMB, Y_SMB = make_training_data("generic/" + game[0] + "/",orient[0], 2)
print(len(X_SMB))
X_KI, Y_KI = make_training_data("generic/" + game[1] + "/",orient[1], 5)
print(len(X_KI))
X_SMB_O, Y_SMB_O = make_training_data("original/" + game[0] + "/",orient[0], 2)
#print(len(X_SMB_O))
X_KI_O, Y_KI_O = make_training_data("original/" + game[1] + "/",orient[1], 5)
#print(len(X_KI_O))
#sys.exit()

X = X_SMB + X_KI
Y = Y_SMB + Y_KI

X_O = X_SMB_O + X_KI_O
Y_O = Y_SMB_O + Y_KI_O

text, smb_text, ki_text, text_o = '', '', '', ''
for x in X:
    text += x
for x in X_O:
    text_o += x
    

chars = sorted(list(set(text)))
print('total chars:', len(chars))
print(chars)

orig_chars = sorted(list(set(text_o)))
print('total chars:', len(orig_chars))
print(orig_chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print(char_indices)
print(indices_char)

orig_char_indices = dict((c, i) for i, c in enumerate(orig_chars))
orig_indices_char = dict((i, c) for i, c in enumerate(orig_chars))
print(orig_char_indices)
print(orig_indices_char)

X_oh = np.zeros((len(X), 16, len(chars)), dtype=np.bool)
for i, x in enumerate(X):
    for t, char in enumerate(x):
        X_oh[i, t, char_indices[char]] = 1

Y_oh = np.array(Y)

RMSprop = RMSprop(lr=0.0005)
SGD = SGD(lr=0.0005)
adadelta = Adadelta(lr=0.0005)
adamax = Adamax(lr=0.0005)
adam = Adam(lr=0.0005)
nadam = Nadam(lr=0.0005)

model = Sequential()
model.add(Dense(256, input_shape=(16, len(chars)), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop, metrics=['accuracy'])
model.fit(X_oh, Y_oh, epochs=50, verbose=1)

model_name = 'generic_classifier'
model.save(model_name + '.h5')
model.save_weights(model_name + '_weights.h5')
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
print('Saved generic classifier to disk.')


X_new = ['---------------X','XXXX--------XXXX','-------------XXX']
X_noh = np.zeros((len(X_new), 16, len(chars)), dtype=np.bool)
for i, x in enumerate(X_new):
    for t, char in enumerate(x):
        X_noh[i, t, char_indices[char]] = 1
Y_new = model.predict_classes(X_noh)
Y_probs = model.predict_proba(X_noh)



X_O_oh = np.zeros((len(X_O), 16, len(orig_chars)), dtype=np.bool)
for i, x in enumerate(X):
    for t, char in enumerate(x):
        X_O_oh[i, t, orig_char_indices[char]] = 1

model_o = Sequential()
model_o.add(Dense(256, input_shape=(16, len(orig_chars)), activation='relu'))
model_o.add(Dropout(0.5))
model_o.add(Dense(256, activation='relu'))
model_o.add(Dropout(0.5))
model_o.add(Flatten())
model_o.add(Dense(1, activation='sigmoid'))
model_o.compile(loss='binary_crossentropy', optimizer=RMSprop, metrics=['accuracy'])
model_o.fit(X_O_oh, Y_oh, epochs=50, verbose=1)


X_new = ['---------------X','####--------####','-------------XXX']
X_O_noh = np.zeros((len(X_new), 16, len(orig_chars)), dtype=np.bool)
for i, x in enumerate(X_new):
    for t, char in enumerate(x):
        X_O_noh[i, t, orig_char_indices[char]] = 1
Y_new = model_o.predict_classes(X_O_noh)
Y_probs = model_o.predict_proba(X_O_noh)

#print(Y_new)
#print(Y_probs)

model_name = 'original_classifier'
model_o.save(model_name + '.h5')
model_o.save_weights(model_name + '_weights.h5')
model_json = model_o.to_json()
with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
print('Saved original classifier to disk.')

scores = model.evaluate(X_oh, Y_oh)
print('Generic model evaluation: ')
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

scores = model_o.evaluate(X_O_oh, Y_oh)
print('Original model evaluation: ')
print('\n%s: %.2f%%' % (model_o.metrics_names[1], scores[1]*100))
