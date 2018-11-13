from __future__ import print_function
from parser import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
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

def shuffle_levels(folder,orientation,duplication = 2):
    levels = []
    for file in os.listdir(folder):
        with open(os.path.join(folder,file),'rb') as infile:
            level = []
            for line in infile:
                level.append(list(str(line.rstrip(),'utf-8')))
            levels.append(level)
    new_levels = []
    for ii in range(duplication):
        new_levels += levels
    levels = new_levels
    print ("Levels: ", levels[0])
    random.shuffle(levels)
    print ("Shuffled: ", levels[0])
    return levels

def parse_levels(levels):
    outstr = ''
    for (level,orientation) in levels:
        width = len(level[0])
        height = len(level)
        outstr += '\n'
        if orientation == 'C':
            for column in range(width):
                outstr += '('
                for row in range(height):
                    outstr += str(level[row][column])
        else:
            for row in range(height):
                outstr += '('
                for column in range(width):
                    outstr += str(level[row][column])
    return outstr

mario_levels = shuffle_levels(blend + "/" + game[0] + "/",orient[0],9)
icarus_levels = shuffle_levels(blend + "/" + game[1] + "/",orient[0],21)
mlo = [(level,'C') for level in mario_levels]
ilo = [(level,'R') for level in icarus_levels]
all_levels = mlo + ilo
random.shuffle(all_levels)
#print (all_levels[0][1])
#print (all_levels[1][1])
#print (all_levels[2][1])
#print (all_levels[3][1])

print (len(all_levels))

text = parse_levels(all_levels)

print(len(text))
#sys.exit()

"""
print("Mario")
mario_text = parse_folder(blend + "/" + game[0] + "/",orient[0],4)
print("Icarus")
icarus_text = parse_folder(blend + "/" + game[1] + "/",orient[1],9)

mario_chars = sorted(list(set(mario_text)))
icarus_chars = sorted(list(set(icarus_text)))
#print (mario_chars)
#print (icarus_chars)
#print('Mario chars:', len(mario_chars))
#print('Icarus chars:', len(icarus_chars))

mario_char_indices = dict((c, i) for i, c in enumerate(mario_chars))
mario_indices_char = dict((i, c) for i, c in enumerate(mario_chars))

icarus_char_indices = dict((c, i) for i, c in enumerate(icarus_chars))
icarus_indices_char = dict((i, c) for i, c in enumerate(icarus_chars))

#print (mario_char_indices)
#print (mario_indices_char)
#print (icarus_char_indices)
#print (icarus_indices_char)

text = mario_text + icarus_text
#print('corpus length:', len(text))
"""
chars = sorted(list(set(text)))
#print('total chars:', len(chars))
#print(chars)

#Make vocabularies
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 46
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
#print(sentences[0], next_chars[0])
#print(sentences[1], next_chars[1])
#sys.exit()

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

"""
mario_sentences = []
mario_next_chars = []
for i in range(0, len(mario_text) - maxlen, step):
    mario_sentences.append(mario_text[i: i + maxlen])
    mario_next_chars.append(mario_text[i + maxlen])
print('nb sequences:', len(mario_sentences))
#print(mario_sentences[0], mario_next_chars[0])
#print(mario_sentences[1], mario_next_chars[1])

X_mario = np.zeros((len(mario_sentences), maxlen, len(mario_chars)), dtype=np.bool)
y_mario = np.zeros((len(mario_sentences), len(mario_chars)), dtype=np.bool)
for i, sentence in enumerate(mario_sentences):
    for t, char in enumerate(sentence):
        X_mario[i, t, mario_char_indices[char]] = 1
    y_mario[i, mario_char_indices[mario_next_chars[i]]] = 1


icarus_sentences = []
icarus_next_chars = []
for i in range(0, len(icarus_text) - maxlen, step):
    icarus_sentences.append(icarus_text[i: i + maxlen])
    icarus_next_chars.append(icarus_text[i + maxlen])
print('nb sequences:', len(icarus_sentences))
#print(icarus_sentences[0], icarus_next_chars[0])
#print(icarus_sentences[1], icarus_next_chars[1])

X_icarus = np.zeros((len(icarus_sentences), maxlen, len(icarus_chars)), dtype=np.bool)
y_icarus = np.zeros((len(icarus_sentences), len(icarus_chars)), dtype=np.bool)
for i, sentence in enumerate(icarus_sentences):
    for t, char in enumerate(sentence):
        X_icarus[i, t, icarus_char_indices[char]] = 1
    y_icarus[i, icarus_char_indices[icarus_next_chars[i]]] = 1
"""
size = 128
layers = 2
dropout = 0.5
learning_rate = 0.005
optimizer = RMSprop(lr=learning_rate)

if weighted:
    ## MARIO LSTM
    model_mario = Sequential()
    #INPUT
    model_mario.add(LSTM(size, input_shape=(maxlen, len(mario_chars)),return_sequences=True))
    model_mario.add(Dropout(dropout))
    #MIDDLE LAYERS
    for ii in range(layers-2):
        model_mario.add(LSTM(size, input_shape=(maxlen,size),return_sequences=True))
        model_mario.add(Dropout(dropout))
    #OUTPUT
    model_mario.add(LSTM(size, input_shape=(maxlen, len(mario_chars)),return_sequences=False))
    model_mario.add(Dropout(dropout))
    model_mario.add(Dense(len(mario_chars)))
    model_mario.add(Activation('softmax'))
    model_mario.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    ## ICARUS LSTM
    model_icarus = Sequential()
    #INPUT
    model_icarus.add(LSTM(size, input_shape=(maxlen, len(icarus_chars)),return_sequences=True))
    model_icarus.add(Dropout(dropout))
    #MIDDLE LAYERS
    for ii in range(layers-2):
        model_icarus.add(LSTM(size, input_shape=(maxlen,size),return_sequences=True))
        model_icarus.add(Dropout(dropout))
    #OUTPUT
    model_icarus.add(LSTM(size, input_shape=(maxlen, len(icarus_chars)),return_sequences=False))
    model_icarus.add(Dropout(dropout))
    model_icarus.add(Dense(len(icarus_chars)))
    model_icarus.add(Activation('softmax'))
    model_icarus.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    # train the model, output generated text after each iteration

    val_loss, val_acc, loss, acc =[], [], [], []
    print("Training Mario")
    for iteration in range(1,101):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        res = model_mario.fit(X_mario, y_mario, batch_size=256, epochs=1, validation_split=0.1)
        cur_loss = res.history['loss'][0]
        val_loss.append(res.history['val_loss'][0])
        val_acc.append(res.history['val_acc'][0])
        loss.append(res.history['loss'][0])
        acc.append(res.history['acc'][0])

    model_name = 'model_shuffled_mario' + '_' + blend + '_weighted'
    model_mario.save(model_name + '.h5')
    model_mario.save_weights(model_name + '_weights.h5')
    model_mario_json = model_mario.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_mario_json)

    temp = open(model_name + "_val_loss.txt", "w")
    for v in val_loss:
        temp.write(str(v) + "\n")
    temp = open(model_name + "_val_acc.txt", "w")
    for v in val_acc:
        temp.write(str(v) + "\n")
    temp = open(model_name + "_loss.txt", "w")
    for v in loss:
        temp.write(str(v) + "\n")
    temp = open(model_name + "_acc.txt", "w")
    for v in acc:
        temp.write(str(v) + "\n")
    print("Saved mario_model to disk")

    val_loss, val_acc, loss, acc =[], [], [], []
    print("Training icarus")
    for iteration in range(1,51):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        res=model_icarus.fit(X_icarus, y_icarus, batch_size=256, epochs=1, validation_split=0.1)
        val_loss.append(res.history['val_loss'][0])
        val_acc.append(res.history['val_acc'][0])
        loss.append(res.history['loss'][0])
        acc.append(res.history['acc'][0])

    model_name = 'model_shuffled_icarus' + '_' + blend + '_weighted'
    model_icarus.save(model_name + '.h5')
    model_icarus.save_weights(model_name + '_weights.h5')

    model_icarus_json = model_icarus.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_icarus_json)

    temp = open(model_name + "_val_loss.txt", "w")
    for v in val_loss:
        temp.write(str(v) + "\n")
    temp = open(model_name + "_val_acc.txt", "w")
    for v in val_acc:
        temp.write(str(v) + "\n")
    temp = open(model_name + "_loss.txt", "w")
    for v in loss:
        temp.write(str(v) + "\n")
    temp = open(model_name + "_acc.txt", "w")
    for v in acc:
        temp.write(str(v) + "\n")

    print("Saved icarus_model to disk")
else:
    model = Sequential()
    #INPUT
    model.add(LSTM(size, input_shape=(maxlen, len(chars)),return_sequences=True))
    model.add(Dropout(dropout))
    #MIDDLE LAYERS
    for ii in range(layers-2):
        model.add(LSTM(size, input_shape=(maxlen,size),return_sequences=True))
        model.add(Dropout(dropout))
    #OUTPUT
    model.add(LSTM(size, input_shape=(maxlen, len(chars)),return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    """
    early_stop = EarlyStopping(monitor='val_loss',min_delta=0.001, patience=5, verbose=1, mode='auto')
    callbacks_list = [early_stop]

    start = time.time()
    res = model.fit(X, y, batch_size=256, epochs=2, callbacks=callbacks_list, validation_split=0.1)
    val_loss = res.history['val_loss']
    val_acc = res.history['val_acc']
    loss = res.history['loss']
    acc = res.history['acc']
    end = time.time()
    """
    iteration = 0
    threshold = 0.001
    val_loss, val_acc, loss, acc = [], [], [], []
    prev_va = float('inf')
    no_change = 0
    best_val_acc = float('-inf')
    print("Training model...")
    #while True:
    for iteration in range(1,51):
        #iteration += 1
        print()
        print('-' * 50)
        print('Iteration', iteration)
        res = model.fit(X,y,batch_size=256, epochs=1, validation_split=0.1)
        vl, va, l, a = res.history['val_loss'][0], res.history['val_acc'][0], res.history['loss'][0], res.history['acc'][0]
        val_loss.append(res.history['val_loss'][0])
        val_acc.append(res.history['val_acc'][0])
        loss.append(res.history['loss'][0])
        acc.append(res.history['acc'][0])
        
        if va > best_val_acc:
            best_val_acc = va
            print("IMPROVED! New best validation acc: ", best_val_acc)
            model_name = 'model_shuffled_combined_' + blend
            model.save(model_name + '.h5')
            model.save_weights(model_name + '_weights.h5')
            model_json = model.to_json()
            with open(model_name + ".json", "w") as json_file:
                json_file.write(model_json)

            temp = open(model_name + "_val_loss.txt", "w")
            for v in val_loss:
                temp.write(str(v) + "\n")
            temp = open(model_name + "_val_acc.txt", "w")
            for v in val_acc:
                temp.write(str(v) + "\n")
            temp = open(model_name + "_loss.txt", "w")
            for v in loss:
                temp.write(str(v) + "\n")
            temp = open(model_name + "_acc.txt", "w")
            for v in acc:
                temp.write(str(v) + "\n")

            print("Saved combined model to disk")

