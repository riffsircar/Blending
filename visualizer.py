import sys
import os
import random
from PIL import Image

"""
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
    """

images = {
    # TODO: Get T, D, M tiles from Icarus

    
    "E": Image.open('Tiles/E.png'),
    "H": Image.open('Tiles/E.png'),
    "G": Image.open('Tiles/G.png'),
    "M": Image.open('Tiles/M.png'),
    "o": Image.open('Tiles/o.png'),
    "S": Image.open('Tiles/S.png'),
    "T": Image.open('Tiles/T.png'),
    "?": Image.open('Tiles/Q.png'),
    "Q": Image.open('Tiles/T.png'),
    "X": Image.open('Tiles/X1.png'),
    "#": Image.open('Tiles/X.png'),
    "-": Image.open('Tiles/-.png'),
    "0": Image.open('Tiles/0.png'),
    "D": Image.open('Tiles/D.png'),
    "<": Image.open('Tiles/PTL.png'),
    ">": Image.open('Tiles/PTR.png'),
    "[": Image.open('Tiles/[.png'),
    "]": Image.open('Tiles/].png')
    }

#layouts = ["scrumpf"]
layouts = ["layout"]
for l in layouts:
    #folder = "levels/" + m
    folder = "layouts/"
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            infile = open(os.path.join(folder,file),'r')
            lines = [line for line in infile]
            rows = len(lines)
            cols = len(lines[0])
            image_width, image_height = cols * 16, rows * 16
            output = Image.new('RGB',(image_width, image_height))
            row, col = 0, 0
            for row, line in enumerate(lines):
                row_string = ''.join(line)
                for col, tile in enumerate(row_string):
                    if tile in images.keys():
                        output.paste(images[tile],(col*16, row*16))
            idx = file.index("gen")
            outfolder = file[idx:idx+4]
            #print(outfolder)
            output.save(os.path.join(folder,file)[:-4] + ".png")
