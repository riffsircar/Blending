from __future__ import print_function
from parser import *
import numpy as np
import random
import sys
import os
import argparse
import json

level_size = 200

def scrumpf(sequences, name):
    layout = []
    if sequences[0][1] == 'C':
        x, y = 0, 184
    else:
        x, y = 0, 199

    prev_seq, prev_orient = None, None
    for sequence in sequences:
        if len(layout) == 0:
            if len(sequence[0]) < 16:
                sequence[0] = ('-' * (16-len(sequence[0]))) + sequence[0]
            layout.append(((x,y), sequence[0]))
            sequence = (sequence[0], sequence[1])
            (prev_seq, prev_orient) = sequence
            continue
        (seq, orient) = sequence
        if len(seq) < 16:
            seq = ('-' * (16-len(seq))) + seq
        if orient == 'C':
            if prev_orient == 'C':
                x += 1
            else:
                x += 15
        else:
            if prev_orient == 'R':
                y -= 1
            else:
                x += 1
                y += 15
        prev_orient = orient
        layout.append(((x,y),seq))

    Xs, Ys = [], []
    for lay in layout:
        (x,y) = lay[0]
        Xs.append(x)
        Ys.append(y)

    x_max, y_min = max(Xs), min(Ys)
    #print(X_max, Y_min)

    layout_fixed = []
    y_max = -1000
    for lay in layout:
        (x,y), seq = lay[0], lay[1]
        y -= y_min
        layout_fixed.append(((x,y),seq))
        if y > y_max:
            y_max = y

    level = [['0' for x in range(x_max+16)] for y in range(y_max+16)]
    #level = [['-' for x in range(level_size)] for y in range(level_size)]

    for i, seq in enumerate(sequences):
        col, row = layout_fixed[i][0]
        s, o = seq[0], seq[1]
        if o == 'C':
            for c, tile in enumerate(s):
                level[row+c][col] = tile
        elif o == 'R':
            for c, tile in enumerate(s):
                level[row][col+c] = tile
        
    
    outfile = open('scrumpf_' + name + '.txt', 'w')
    outfile.write('\n'.join([''.join([tile for tile in row]) for row in level]))
    outfile.close()
    
        

def layout(sequences, name):
    #print("*************************INSIDE LAYOUT*****************************")
    layout = []
    #print(sequences)
    if sequences[0][1] == 'C':
        x, y = 0, 184
    else:
        x, y = 0, 199

    for (i,seq) in enumerate(sequences):
        if '\n' in seq[0]:
            temp = seq[0].replace('\n','')
            sequences[i] = (temp,seq[1])

    for (i,seq) in enumerate(sequences):
        if len(seq[0]) < 16:
            temp = ('-' * (16-len(seq[0]))) + seq[0]
            sequences[i] = (temp,seq[1])
        elif len(seq[0]) > 16:
            temp = seq[0][:16]
            sequences[i] = (temp,seq[1])

    prev_seq, prev_orient = None, None
    prev_solid_y_row = None
    for sequence in sequences:
        if len(layout) == 0:
            if len(sequence[0]) < 16:
                sequence[0] = ('-' * (16-len(sequence[0]))) + sequence[0]
            layout.append(((x,y), sequence[0],sequence[1]))
            sequence = (sequence[0], sequence[1])
            (prev_seq, prev_orient) = sequence
            continue
        (seq, orient) = sequence
        if len(seq) < 16:
            seq = ('-' * (16-len(seq))) + seq

        if prev_orient == 'C':
            if orient == 'R':
                ground = [prev_seq.find(i) for i in "XEQ?<>[]oSMT" if i in prev_seq]
                if ground:
                    #print(prev_seq, ground)
                    #print(min(ground))
                    y += min(ground)
                    prev_solid_y_row = y
                    #print("Prev solid: ", prev_solid_y_row)
            else:
                ground = [prev_seq.find(i) for i in "XEQ?<>[]oSMT" if i in prev_seq]
                if ground:
                    prev_solid_y_row = y
            x += 1
            layout.append(((x,y),seq,orient))
        else:
            if orient == 'C':
                x += 16
                ground = [seq.find(i) for i in "XEQ?<>[]oSMT" if i in seq]
                if ground:
                    #print(seq, ground)
                    #print(min(ground))
                    y -= min(ground)
                    if prev_solid_y_row:
                        #print("Prev solid y row: ", prev_solid_y_row)
                        y = prev_solid_y_row - min(ground)
            else:
                y -= 1
                ground = [prev_seq.find(i) for i in "XEQ?<>[]oSMT" if i in prev_seq]
                if ground:
                    prev_solid_y_row = y
            layout.append(((x,y),seq,orient))
        prev_seq, prev_orient = seq, orient

    Xs, Ys = [], []
    for lay in layout:
        (x,y) = lay[0]
        Xs.append(x)
        Ys.append(y)

    x_max, y_min = max(Xs), min(Ys)
    #print(X_max, Y_min)

    layout_fixed = []
    y_max = -1000
    for lay in layout:
        (x,y), seq, orient = lay[0], lay[1], lay[2]
        y -= y_min
        layout_fixed.append(((x,y),seq))
        if orient == 'C':
            if (y+16) > y_max:
                y_max = (y+16)
        else:
            if y > y_max:
                y_max = y

    """
    for lay in layout_fixed:
        print(lay[0], lay[1])
    """
    
    level = [['0' for x in range(x_max+16)] for y in range(y_max+1)]
    #level = [['-' for x in range(level_size)] for y in range(level_size)]

    for i, seq in enumerate(sequences):
        col, row = layout_fixed[i][0]
        #print(seq, col, row)
        s, o = seq[0], seq[1]
        #print(s,o)
        try:
            if o == 'C':
                for c, tile in enumerate(s):
                    level[row+c][col] = tile
            elif o == 'R':
                for c, tile in enumerate(s):
                    level[row][col+c] = tile
        except:
            print(row, col, c)
            print(x_max, y_max)
            print("Seq: ", seq)
            print("Layout: ", i, layout_fixed[i][0], row, col, c)
            for i, seq in enumerate(sequences):
                print(i, seq)
            sys.exit()
        
    
    outfile = open('layout_' + name + '.txt', 'w')
    outfile.write('\n'.join([''.join([tile for tile in row]) for row in level]))
    outfile.close()
        
"""
def layout_level(sequences, name):
    layout = []
    if sequences[0][1] == 'C':
        x, y = 0, 184
        last_x = sequences[0][1].find('X')
        last_s = sequences[0][1].find('S')
        last_q = sequences[0][1].find('?')
        last_Q = sequences[0][1].find('Q')
        last_p = sequences[0][1].find('<')
    else:
        x, y = 0, 199
        if blend == 'generic':
            last_ground = sequences[0][1].find('X')
        else:
            last_ground = sequences[0][1].find('#')
    #layout.append((x,y))
    
    for i, sequence in enumerate(sequences):
        if i == 0:
            seq, orient = sequence[0], sequence[1]
            layout.append((x,y))
            continue
        new_seq, new_orient = sequence[0], sequence[1]
        if orient == 'C':
            idx = seq.find('X')
            if idx != -1:
                last_ground = idx
            if new_orient == 'C':
                x += 1
            elif new_orient == 'R':
                x += 1
                if idx != -1:
                    y += idx
                else:
                    y += last_ground
        elif orient == 'R':
            if new_orient == 'C':
                idx = new_seq.rfind('X')
                x += 16
                if idx != -1:
                    y -= idx
                else:
                    y -= 15
            elif new_orient == 'R':
                y -= 1
        layout.append((x,y))
        seq, orient = new_seq, new_orient

    level = [['0' for x in range(level_size)] for y in range(level_size)]

    for i, seq in enumerate(sequences):
        col, row = layout[i]
        #print(seq, col, row)
        s, o = seq[0], seq[1]
        if o == 'C':
            for c, tile in enumerate(s):
                level[row+c][col] = tile
        elif o == 'R':
            for c, tile in enumerate(s):
                level[row][col+c] = tile

    outfile = open('layout_' + name + '.txt', 'w')
    outfile.write('\n'.join([''.join([tile for tile in row]) for row in level]))
    outfile.close()

def layout_level(sequences, name):
    layout = []
    if sequences[0][1] == 'C':
        x, y = 0, 184
        last_x = sequences[0][1].find('X')
        last_s = sequences[0][1].find('S')
        last_q = sequences[0][1].find('?')
        last_Q = sequences[0][1].find('Q')
        last_p = sequences[0][1].find('<')
    else:
        x, y = 0, 199
        if blend == 'generic':
            last_ground = sequences[0][1].find('X')
        else:
            last_ground = sequences[0][1].find('#')
    #layout.append((x,y))
    
    for i, sequence in enumerate(sequences):
        if i == 0:
            seq, orient = sequence[0], sequence[1]
            layout.append((x,y))
            continue
        new_seq, new_orient = sequence[0], sequence[1]
        if orient == 'C':
            idx = seq.find('X')
            if idx != -1:
                last_ground = idx
            if new_orient == 'C':
                x += 1
            elif new_orient == 'R':
                x += 1
                if idx != -1:
                    y += idx
                else:
                    y += last_ground
        elif orient == 'R':
            if new_orient == 'C':
                idx = new_seq.rfind('X')
                x += 16
                if idx != -1:
                    y -= idx
                else:
                    y -= 15
            elif new_orient == 'R':
                y -= 1
        layout.append((x,y))
        seq, orient = new_seq, new_orient

    level = [['0' for x in range(level_size)] for y in range(level_size)]

    for i, seq in enumerate(sequences):
        col, row = layout[i]
        #print(seq, col, row)
        s, o = seq[0], seq[1]
        if o == 'C':
            for c, tile in enumerate(s):
                level[row+c][col] = tile
        elif o == 'R':
            for c, tile in enumerate(s):
                level[row][col+c] = tile

    outfile = open(name + '.txt', 'w')
    outfile.write('\n'.join([''.join([tile for tile in row]) for row in level]))
    outfile.close()
"""

#generators = ["gen1","gen2","gen3","gen4","gen5","gen6","gen7","gen8"]
generators = ["gen9","gen10"]

for g in generators:
    folder = "levels/" + g
    for file in os.listdir(folder):
        infile = open(os.path.join(folder,file),'r')
        sequences = json.load(infile)['sequences']
        #print(file[:-5])
        #scrumpf(sequences, g + '_' + file[:-5])
        layout(sequences, g + '_' + file[:-5])
        #sys.exit()
