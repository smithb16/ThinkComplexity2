## Module import and boilerplate
""" Code example from Complexity and Computation, a book about
exploring complexity science with Python.  Available free from

http://greenteapress.com/complexity

Copyright 2016 Allen Downey
MIT License: http://opensource.org/licenses/MIT
"""
from __future__ import print_function, division
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib import animation
from collections import deque

"""
For animation to work in the notebook, you might have to install
ffmpeg.  On Ubuntu and Linux Mint, the following should work.

    sudo add-apt-repository ppa:mc3man/trusty-media
    sudo apt-get update
    sudo apt-get install ffmpeg
"""

from Cell2D import Cell2D
from scipy.signal import correlate2d

## Class and function definition here
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

    def count_changing(self):
        """Count the number of cells that will change with the next step."""
        c = correlate2d(self.array, self.kernel, mode='same')
        flipped = np.sum(self.array != self.table[c])
        return flipped

    def start_random(self, p=0.1):
        """Starts cells from a random state. Cells turned on with probability p.
        """
        m, n = self.array.shape
        self.array = np.random.choice([0,1], size=(m,n), p=[(1-p),p])

    def run(self, in_a_row=20):
        """Run until 10 frames in a row have same number of cells. Used to claim
        stability. Also claims stability for 2-value oscillators.

        in_a_row: int of number of same frames in a row to claim stability
        oscillators: number of oscillating cells in a steady-state.
                    This value can be subtracted to provide a more accurate
                    assessment of number of affected cells.
        returns: number of steps to stabilize
            returns -1 if system does not stabilize or exhibit 2-frame oscillation"""

        m, n = self.array.shape
        affected = np.zeros((m,n),dtype=np.bool)
        counts = deque(range(in_a_row))

        for i in itertools.count(1):
            a = self.array

            #print('i:\n', i)
            #print('counts:\n', counts)
            #print('len(set(counts)):\n', len(set(counts)))

            c = correlate2d(a, self.kernel, mode='same')
            affected += (a != self.table[c])

            self.step()
            counts.append(np.sum(a))
            counts.popleft()

            if len(set(counts)) <=2:
                s = np.sum(affected)
                t = i - in_a_row + 1
                return t, s

            if i >= 4000:
                print('Does not converge')
                return -1, -1

    def flip_random(self, iters=1):
        """Flips a random cell."""
        m, n = self.array.shape
        for _ in range(iters):
            row = np.random.randint(0,m)
            col = np.random.randint(0,n)
            self.array[row, col] = (self.array[row, col] + 1) % 2

    def flip_and_run(self, iters=1):
        """Flip a random cell and run until stability.
        returns: duration, affected
        """
        self.flip_random(iters)
        duration, affected = self.run()
        if duration == -1:
            duration, affected = self.flip_and_run(iters)

        return duration, affected

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

    #life = HighLife(m, n)
    #cwd = os.getcwd()
    #filename = 'replicator.rle'
    #fin = cwd + '/' + filename

    #life.start_from_rle(fin)
    ## Make random
    #m = 30
    #life = Life(m)
    #life.start_random(p=0.4)
    #life.run()

    ### Now drop random cell and re-animate
    #for i in range(100):
    #    life.flip_random(5)
    #    time_steps, affected = life.run()
    #    print('time steps: ', time_steps, end='   ')
    #    print('affected : ', affected )


    #Make puffer train
    #col = 120
    #life.add_cells(n//2+12, col, *lwss)
    #life.add_cells(n//2+26, col, *lwss)
    #life.add_cells(n//2+19, col, *bhep)

    #life.start_random()
    #fig = life.draw()
    #ani = life.animate()
    #plt.show(block=True)

    ## Generate sand pile and test many avalanches
    life = Life(m=30)
    life.start_random()
    life.run()
    iters=1000000
    flip_cells = 5
    res = [life.flip_and_run(flip_cells) for _ in range(iters)]

    T, S = np.transpose(res)
    T = T[T>1]
    S = S[S>flip_cells]

    ## Make and plot PMFs
    pmfT = Pmf.from_seq(T)
    pmfS = Pmf.from_seq(S)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    pmfT.plot(label='T')
    decorate(xlabel='GoL Cascade duration',
                     ylabel='PMF',
                     xlim=[1, 500], loc='upper right')

    plt.subplot(1, 2, 2)
    pmfS.plot(label='S')
    decorate(xlabel='Cells affected',
                     xlim=[1, 800])

    savefig('myfigs/chap08-12')
    plt.show(block=True)

    ## Make PMF & CDF  on log-log
    plt.figure(figsize=(10, 10))

    # Subplot1 - PMF of T
    plt.subplot(2, 2, 1)

    #xs = [2, 600]
    #ys = [1.3e-1, 2.2e-4]
    #print(slope(xs, ys))

    options = dict(color='gray', alpha=0.3, linewidth=4)
    #plt.plot(xs, ys, **options)

    pmfT.plot(label='T', linewidth=2)
    decorate(xlabel='GoL Cascade duration',
                     xlim=[1, 500],
                     ylabel='PMF',
                     xscale='log',
                     yscale='log',
                     loc='upper right')

    # Subplot 2 - PMF of S
    plt.subplot(2, 2, 2)

    #xs = [1, 5000]
    #ys = [1.3e-1, 2.3e-5]
    #print(slope(xs, ys))

    #plt.plot(xs, ys, **options)
    pmfS.plot(label='S', linewidth=1)
    decorate(xlabel='Cells affected',
                     xlim=[1, 800],
                     xscale='log',
                     yscale='log')

    # Make CDFs
    cdfS = Cdf.from_seq(S)
    cdfT = Cdf.from_seq(T)

    # Subplot 3 - CDF T
    plt.subplot(2,2,3)
    (1-cdfT).plot(color='C0', label='T')
    decorate(xlabel='GoL cascade duration',xscale='log',
            ylabel='CCDF', yscale='log')

    # Subplot 4 - CDF S
    plt.subplot(2,2,4)
    (1-cdfS).plot(color='C0', label='S')
    decorate(xlabel='Cells affected',xscale='log',
            ylabel='CCDF', yscale='log')


    savefig('myfigs/chap08-13')
    plt.show(block=True)

## Ending block
if __name__ == '__main__':
    main(*sys.argv)

