
""" Code example from Complexity and Computation, a book about
exploring complexity science with Python.  Available free from

http://greenteapress.com/complexity

Copyright 2016 Allen Downey
MIT License: http://opensource.org/licenses/MIT
"""
from __future__ import print_function, division
import re
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
For animation to work in the notebook, you might have to install
ffmpeg.  On Ubuntu and Linux Mint, the following should work.

    sudo add-apt-repository ppa:mc3man/trusty-media
    sudo apt-get update
    sudo apt-get install ffmpeg
"""

from Cell2D import Cell2D
from scipy.signal import correlate2d

class Life(Cell2D):
    """Implementation of Conway's Game of Life."""
    kernel = np.array([[1, 1, 1],
                       [1,10, 1],
                       [1, 1, 1]])

    table = np.zeros(20, dtype=np.uint8)
    table[[3, 12, 13]] = 1

    def step(self):
        """Executes one time step."""
        c = correlate2d(self.array, self.kernel, mode='same')
        self.array = self.table[c]

    def start_random(self, p=0.1):
        """Starts cells from a random state. Cells turned on with probability p.
        """
        m, n = self.array.shape
        self.array = np.random.choice([0,1], size=(m,n), p=[(1-p),p])

    def start_from_rle(self, filename):
        """Adds cells based on RLE file data.
        filename: name of RLE file.
        """
        m, n = self.array.shape
        rle_str = rle_to_string(filename)
        s = decode_rle(rle_str)
        t = string_to_list(s)
        rows = len(t)
        cols = max( len(i) for i in t )
        self.add_cells((m - rows) // 2, (n - cols) // 2, *t)

class HighLife(Life):
    """Implementation of HighLife based on Game of Life"""
    kernel = np.array([[1, 1, 1],
                       [1,10, 1],
                       [1, 1, 1]])

    table = np.zeros(20, dtype=np.uint8)
    table[[3, 6, 12, 13]] = 1

def rle_to_string(filename):
    fin = open(filename)
    s = ''
    for line in fin:
        if ('#' not in line) and ('x' not in line):
            s += line.strip()

    s = s.split('!')[0]
    return s

def decode_rle(s):
    """Decodes RLE text to replace integer multipliers with their intended letter.
    s: string of rle text
    returns - new: string of expanded text
    """
    new = ''
    while s:
        if re.search('^\d+',s):
            val = re.search('^\d+',s).group()
            mult = int(val)
            s = s.lstrip(val)
        else:
            mult = 1

        c = s[0]
        new += c * mult
        s = s.lstrip(c)

    return new

def string_to_list(s):
    """Converts a decoded rle string to a list of strings of binary.
    """
    d = {'b':'0','o':'1'}
    t = []
    row = '0'

    for c in s:
        if c == '$':
            t.append(row)
            row = '0'
        else:
            row += d[c]
    t.append(row)
    return t

def main(script, *args):
    """Constructs a puffer train.

    Uses the entities in this file:
    http://www.radicaleye.com/lifepage/patterns/puftrain.lif
    """

# Make puffer train and animate.
    lwss = [
        '0001',
        '00001',
        '10001',
        '01111'
    ]

    bhep = [
        '1',
        '011',
        '001',
        '001',
        '01'
    ]

    m = 100
    n = 100
    life = HighLife(m, n)
    cwd = os.getcwd()
    filename = 'replicator.rle'
    fin = cwd + '/' + filename

    life.start_from_rle(fin)
    ani = life.animate()
    plt.show(block=True)

    #Make puffer train
    #col = 120
    #life.add_cells(n//2+12, col, *lwss)
    #life.add_cells(n//2+26, col, *lwss)
    #life.add_cells(n//2+19, col, *bhep)

    #life.start_random()
    #fig = life.draw()
    #ani = life.animate()
    #plt.show(block=True)

if __name__ == '__main__':
    main(*sys.argv)

