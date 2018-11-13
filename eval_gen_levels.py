from __future__ import print_function
from parser import *
import numpy as np
import random
import sys
import time
import os
import argparse
import json

levels = []

patterns_smb = []
patterns_ki = []
orientation = 'C'
for file in os.listdir("Generic/SMB/"):
    with open(os.path.join("Generic/SMB/",file),'rb') as infile:
        level = []
        for line in infile:
            level.append(list(str(line.rstrip(),'utf-8')))
            #level.append(str(line.rstrip(),'utf-8'))
        levels.append(level)

print(len(levels))

level_strs = []
for level in levels:
    level_str = []
    width = len(level[0])
    height = len(level)
    for column in range(width):
        line_str = ''
        for row in range(height):
            line_str += str(level[row][column])
        if line_str not in patterns_smb:
            patterns_smb.append(line_str)
        level_str.append(line_str)
    level_strs.append(level_str)

levels = []

for file in os.listdir("Generic/KI/"):
    with open(os.path.join("Generic/KI/",file),'rb') as infile:
        level = []
        for line in infile:
            level.append(list(str(line.rstrip(),'utf-8')))
            #level.append(str(line.rstrip(),'utf-8'))
        levels.append(level)

print(len(levels))


for level in levels:
    level_str = []
    width = len(level[0])
    height = len(level)
    for row in range(height):
        line_str = ''
        for column in range(width):
            line_str += str(level[row][column])
        if line_str not in patterns_ki:
            patterns_ki.append(line_str)
        level_str.append(line_str)
    level_strs.append(level_str)


print (len(patterns_smb))
print (len(patterns_ki))

def leniency(level):
    e, g = 0, 0
    for seq in level:
        e += seq.count('E')
        if seq.count('-') == len(seq):
            g += 0.5
    return((e+g)/len(level))

def density(level):
    total = 0
    for seq in level:
        total += seq.count('X')
        total += seq.count('S')
        total += seq.count('Q')
        total += seq.count('?')
    return(total/len(level))

def pattern_density(level,patterns):
    total = 0
    for seq in level:
        if seq in patterns:
            total += 1
    return(total/len(level))

def pattern_variation(level, patterns):
    variety = []
    for seq in level:
        if seq in patterns:
            if seq not in variety:
                variety.append(seq)

    return(len(variety)/len(level))


def calculate_metrics(level):
    global patterns_smb, patterns_ki
    leniency, d, pd, pv = 0, 0, 0, 0
    e, g = 0, 0
    variety = []
    seqs = len(level)
    for lev in level:
        seq, orient = lev[0], lev[1]
        e -= seq.count('E')
        if seq.count('-') == len(seq) and orient == 'C':
            g -= 0.5
        d += seq.count('X')+seq.count('S')+seq.count('Q')+seq.count('?')
        if orient == 'C':
            if seq in patterns_smb:
                pd += 1
                if seq not in variety:
                    variety.append(seq)
        else:
            if seq in patterns_ki:
                pd += 1
                if seq not in variety:
                    variety.append(seq)

    leniency = (e+g)/seqs
    d /= seqs
    pd /= seqs
    pv = len(variety)/seqs
    return (leniency, d, pd, pv)

generators = ["gen1","gen2","gen3","gen4","gen5","gen6","gen7","gen8","gen9","gen10"]


for g in generators:
    folder = "levels/" + g
    print(folder)
    res_file = open(g + ".csv","w")
    res_file.write("Level,Leniency,Density,PD,PV\n")
    for file in os.listdir(folder):
        infile = open(os.path.join(folder,file),'r')
        sequences = json.load(infile)['sequences']
        #print(len(sequences))
        leniency, d, pd, pv = calculate_metrics(sequences)
        res_file.write(file + "," + str(leniency) + "," + str(d) + "," + str(pd) + "," + str(pv) + "\n")
    res_file.close()
