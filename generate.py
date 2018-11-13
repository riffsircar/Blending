from __future__ import print_function
from parser import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.models import model_from_json
import numpy as np
import random
import sys
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1)
parser.add_argument("--sh", action='store_true')
parser.add_argument("--w", action='store_true')
parser.add_argument("--sep", action='store_true')
parser.add_argument("-b", default="generic")
parser.add_argument("-seg", type=int, default=20)
parser.add_argument("-size", type=int, default=200)
parser.add_argument("-seed", default="SMB")
args = parser.parse_args()

num_levels = args.n
shuffled = args.sh
weighted = args.w
blend = args.b
segment_size = args.seg
level_size = args.size
seed = args.seed
separated = args.sep

print(num_levels)
print(shuffled)
print(weighted)
print(blend)
print(segment_size)
print(level_size)
print(seed)
print(separated)


print("Blend: ", blend, "\tWeighted: ", str(weighted), "\tNum Levels: ", str(num_levels))



def parse_folder(folder,orientation,duplication = 2):
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
    random.shuffle(levels)

    outstr = ''
    for level in levels:
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

mario_text = parse_folder(blend + "/SMB/",'C',1)
icarus_text = parse_folder(blend + "/KI/",'R',1)

mario_chars = sorted(list(set(mario_text)))
icarus_chars = sorted(list(set(icarus_text)))

mario_char_indices = dict((c, i) for i, c in enumerate(mario_chars))
mario_indices_char = dict((i, c) for i, c in enumerate(mario_chars))

icarus_char_indices = dict((c, i) for i, c in enumerate(icarus_chars))
icarus_indices_char = dict((i, c) for i, c in enumerate(icarus_chars))

if seed == 'SMB':
    text = mario_text + icarus_text
else:
    text = icarus_text + mario_text
    
chars = sorted(list(set(text)))


#Make vocabularies
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

class_chars = sorted(list(set(text)))
class_chars.remove('(')
class_chars.remove('\n')
class_char_indices = dict((c, i) for i, c in enumerate(class_chars))
indices_class_char = dict((i, c) for i, c in enumerate(class_chars))


maxlen = 46
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
#print('nb sequences:', len(sentences))

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

mario_sentences = []
mario_next_chars = []
for i in range(0, len(mario_text) - maxlen, step):
    mario_sentences.append(mario_text[i: i + maxlen])
    mario_next_chars.append(mario_text[i + maxlen])
#print('nb sequences:', len(mario_sentences))

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
#print('nb sequences:', len(icarus_sentences))

X_icarus = np.zeros((len(icarus_sentences), maxlen, len(icarus_chars)), dtype=np.bool)
y_icarus = np.zeros((len(icarus_sentences), len(icarus_chars)), dtype=np.bool)
for i, sentence in enumerate(icarus_sentences):
    for t, char in enumerate(sentence):
        X_icarus[i, t, icarus_char_indices[char]] = 1
    y_icarus[i, icarus_char_indices[icarus_next_chars[i]]] = 1


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    if temperature == 0.0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


folder = '2x128_large_shuffled' if shuffled else '2x128_large'

#model_combined, model_mario, model_icarus = None, None, None
if not shuffled and separated:
    json_file = open(folder + '/model_mario_' + blend + '_weighted.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_mario = model_from_json(loaded_model_json)
    model_mario.load_weights(folder + '/model_mario_' + blend + '_weighted_weights.h5')
    print('Loaded mario model from disk...')
    json_file = open(folder + '/model_icarus_' + blend + '_weighted.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_icarus = model_from_json(loaded_model_json)
    model_icarus.load_weights(folder + '/model_icarus_' + blend + '_weighted_weights.h5')
    print('Loaded icarus model from disk...')
    
json_file = open(folder + '/model_combined_' + blend + '.json','r')
loaded_model_json = json_file.read()
json_file.close()
model_combined = model_from_json(loaded_model_json)
model_combined.load_weights(folder + '/model_combined_' + blend + '_weights.h5')
print('Loaded combined model from disk...')


json_file = open('classifiers/' + blend + '_classifier.json','r')
loaded_model = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model)
classifier.load_weights('classifiers/' + blend + '_classifier_weights.h5')
print('Classifier loaded.')

weight = [0.1, 0.9]
#mm = load_model('Model_Mario.h5')
if not weighted:
    print("Using seed: ", seed)
    
for level_num in range(num_levels):
    print("Generating level " + str(level_num) + "...")
    if weighted:
        generated = ''
        start_index = 0
        diversity = 1.0

        num_segments = level_size/segment_size
        seg_num = 1
        seq_or = []
        level = []
        while seg_num <= num_segments:
            choice = np.random.rand()
            game = "Mario" if choice < weight[0] else "Icarus"
            orient = ""
            print("Seg Num: ", seg_num)
            print("Generating segment for ", game)
            generated = ''

            if not separated:
                sentence = mario_text[start_index:start_index+maxlen] if game == "Mario" else icarus_text[start_index:start_index+maxlen]
                generated = ''
                generated += sentence
                while len(generated) <= (segment_size * 17):
                    #print("Inside len gen loop")
                    x = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x[0, t, char_indices[char]] = 1.
                    preds = model_combined.predict(x, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = indices_char[next_index]
                    #print("SENTENCE: ", sentence)
                    if next_char == '(' and (len(generated)-generated.rfind('(')) >= 17:
                        stuck_count = 0
                        seq = generated[generated.rfind('(')+1:]
                        seq = seq[:16]
                        seq_oh = np.zeros((1, len(seq), len(class_chars)), dtype=np.bool)
                        for t, char in enumerate(seq):
                            seq_oh[0, t, class_char_indices[char]] = 1
                            y = classifier.predict_classes(seq_oh)
                            prob = classifier.predict_proba(seq_oh)

                        if prob[0][0] >= 0.45 and prob[0][0] <= 0.55:
                            if orient == "":
                                if np.random.rand() < 0.5:
                                    orient = "Mario"
                                else:
                                    orient = "Icarus"
                        elif y[0][0] == 0:
                            orient = "Mario"
                        else:
                            orient = "Icarus"

                        if orient == game:
                            generated += next_char
                            sentence = sentence[1:]+next_char
                        else:
                            generated = generated[:generated.rfind('(')+1]
                            sentence = generated[-maxlen:]
                    elif next_char != '(':
                        stuck_count = 0
                        generated += next_char
                        sentence = sentence[1:]+next_char
                    else:
                        stuck_count += 1
                        print("PUTAIN!!", stuck_count, generated, next_char)
                        if stuck_count >= 50:
                            stuck_count = 0
                            sentence = mario_text[start_index:start_index+maxlen] if game == "Mario" else icarus_text[start_index:start_index+maxlen]
                            generated = ''
                            generated += sentence
            else:
                if game == "Mario":
                    sentence = mario_text[start_index: start_index + maxlen]
                    generated += sentence
                    print('----- Generating with seed: "' + sentence + '"')
                    while len(generated) < (segment_size * 17):
                        x = np.zeros((1, maxlen, len(mario_chars)))
                        for t, char in enumerate(sentence):
                            x[0, t, mario_char_indices[char]] = 1.
                        preds = model_mario.predict(x, verbose=0)[0]
                        next_index = sample(preds, diversity)
                        next_char = mario_indices_char[next_index]
                        generated += next_char
                        sentence = sentence[1:] + next_char
                else:
                    sentence = icarus_text[start_index: start_index + maxlen]
                    generated += sentence
                    print('----- Generating with seed: "' + sentence + '"')
                    while len(generated) < (segment_size * 17):
                        x = np.zeros((1, maxlen, len(icarus_chars)))
                        for t, char in enumerate(sentence):
                            x[0, t, icarus_char_indices[char]] = 1.
                        preds = model_icarus.predict(x, verbose=0)[0]
                        next_index = sample(preds, diversity)
                        next_char = icarus_indices_char[next_index]
                        generated += next_char
                        sentence = sentence[1:] + next_char
                
            print("Segment number ", str(seg_num), ": ", generated)
            seg_num += 1
            #level += generated
            sequences = generated.split('(')[1:]
            orient = "C" if game == "Mario" else "R"
            seq_or = []
            for seq in sequences:
                seq_or.append((seq,orient))
            level = level + seq_or
        print(level)

        data = {}

        data['sequences'] = []
        for so in level:
            #print(so)
            data['sequences'].append(so)

        print("Data: ", data)

        level_name = 'weighted_w1_' + str(weight[0]) + '_w2_' + str(weight[1]) + '_level_' + str(level_num) + '_' + blend
        if shuffled:
            level_name += '_shuffled.json'
        if separated:
            level_name += '_separated.json'
        else:
            level_name += '.json'
        with open(level_name, 'w') as outfile:
            json.dump(data,outfile)
        outfile.close()
        
    else:
        start_index = 0
        diversity = 1.0
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        while len(generated) <= (level_size * 17):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            preds = model_combined.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char

        sequences = generated.split('(')[1:]
        while '' in sequences:
            sequences.remove('')

        print(class_chars, len(class_chars))
        seq_or = []
        prev = None
        game = ""
        for seq in sequences:
            if len(seq) >= 17:
                print(seq, len(seq))
                seq = seq[:16]
            if len(seq) < 16:
                print(seq,len(seq))
                seq = ('-' * (16-len(seq))) + seq
            seq_oh = np.zeros((1, len(seq), len(class_chars)), dtype=np.bool)
            for t, char in enumerate(seq):
                seq_oh[0, t, class_char_indices[char]] = 1
                y = classifier.predict_classes(seq_oh)
                prob = classifier.predict_proba(seq_oh)
            
            if prob[0][0] >= 0.45 and prob[0][0] <= 0.55:
                if game == "":
                    if np.random.rand() < 0.5:
                        game = "C"
                    else:
                        game = "R"
            elif y[0][0] == 0:
                game = "C"
            else:
                game = "R"

            seq_or.append([seq,game])

        data = {}

        data['sequences'] = []
        for so in seq_or:
            #print(so)
            data['sequences'].append(so)

        print("Data: ", data)

        level_name = 'level_' + str(level_num) + '_' + blend + '_combined_seed_' + seed
        if shuffled:
            level_name += '_shuffled.json'
        else:
            level_name += '.json'
        with open(level_name, 'w') as outfile:
            json.dump(data,outfile)
        outfile.close()
