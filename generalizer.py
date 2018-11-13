import sys
import os
import random


folder = sys.argv[1]

generic_mapping = {
    # already generic
    "-": "-",
    
    # mario to generic
    "X": "X",
    "E": "E",
    "Q": "Q",
    "?": "?",
    "<": "<",
    ">": ">",
    "[": "[",
    "]": "]",
    "o": "o",
    "S": "S",

    # icarus to generic
    "#": "X",
    "H": "E",
    "T": "T",
    "M": "M",
    "D": "D"
    }



levels = []
for file in os.listdir(folder):
    with open(os.path.join(folder,file),'rb') as infile:
        level = []
        for line in infile:
            level.append(list(str(line.rstrip())))
        #level.reverse()

    #print(level)
    #sys.exit()
            
    new_level = ''
    outfile = open("Generic/" + file,"w")
    for l in level:
        #outfile.write(''.join(l))
        temp = ''.join(l)
        #print(temp)
        for c in temp:
            if c in generic_mapping.keys():
                new_level += generic_mapping[c]
            else:
                new_level += '-'
        new_level += '\n'
    #print("New: ", new_level)  
    outfile.write(new_level)
    outfile.close()

