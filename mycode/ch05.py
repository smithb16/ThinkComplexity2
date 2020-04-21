## Import modules and boilerplate
import random
import numpy as np
import matplotlib.pyplot as plt

from Cell1D import *
from scipy.signal import correlate2d

## Function definitions

class Turing(Cell1D):
    """Represents a 1-D Turing machine"""
    def __init__(self, table, n, m=None):
        """Initialize the Turing machine.

        Attributes:

        """
        self.n = n
        self.states = ['A']
        self.state = self.states[0]
        self.m = 2*n + 1 if m is None else m
        self.tape = np.zeros((n, self.m), dtype=np.uint8)
        self.head = np.zeros(n, dtype=np.int64)
        self.head[0] = self.m//2
        self.row = 0

    def draw(self, start=0, end=None):
        """Draw a Turing machine with read head"""

        # Remove columns of zeros at left and right, leaving a border
        border = 2
        filled = np.argwhere(np.any(self.tape !=0, axis=0))
        if start == 0:
            start = max( min(filled)-border, 0)
            start = int(start)
        if end == None:
            end = min( max(filled) + border+1, self.m-1)
            end = int(end)
        a = self.tape[:, start:end]

        # Plot Turing machine
        fig = plt.figure()
        plt.imshow(a, cmap='Blues', alpha=0.7)

        # Add read head to plot
        x = self.head - start
        y = np.arange(self.n)
        plt.plot(x,y,'r.')

        plt.xticks([])
        plt.yticks([])
        return fig

    def step(self):
        """Executes one time step by computing the next row of the array and
        moving the read head."""

        i = self.row
        pos = self.head[i]
        val = self.tape[i][pos]
        new_val, move, new_state = table[val, self.state]

        self.tape[i+1] = self.tape[i]
        self.tape[i+1][pos] = new_val
        self.state = new_state
        self.states.append(new_state)

        if move == 'L':
            self.head[i+1] = pos - 1
        else:
            self.head[i+1] = pos + 1

        self.row += 1

def my_correlate(row, window, mode='constant'):
    """Correlates 'row' with 'window'.
    row: np.array with one row
    window: np.array with one row
    mode:
        'constant', leaves row unchanged, thereby shortening
             returned row by len(window)-1
        'same': pads input row to keep returned row of same size
            uneven padding of size 'n' is applied n//2 to start of 'a'
            (n-1)//2 to tail of 'a'
    return: np.array with one row
    """
    if mode == 'same':
        pl = len(window)//2
        pr = (len(window)-1)//2
        row = np.pad(row, (pl, pr))
    cols = len(row)
    N = len(window)
    c = [c_k(row, window, k) for k in range(cols-N + 1)]
    return np.array(c)

def c_k(a,w,k):
    """Convolves window beginning at k-th element of 'a' with windows"""
    N = len(w)
    return sum(a[k:k+N] * w)

def iterate_ca(rule, n=100):

    i=0
    while True:
        i += 1
        s = '{0:b}'.format(i)
        fig = plt.figure(figsize=(11,7))
        plt.xlabel( s + ' ' + str(i) )

        ca = Cell1D(rule, n)
        ca.start_string(s)
        ca.loop(n-1)
        ca.draw()

        fig.tight_layout()
        plt.show(block=True)
        t = input('press <Enter> or "q" to quit\n')
        if t == 'q': break

def iterate_ca_grid(rule, bits=4, n=100):


    fig_row = 8
    fig_col = 2**bits / fig_row
    fig = plt.figure(figsize=(11,7))

    for i in range(2**bits):
        s = '{0:b}'.format(i)
        fig.add_subplot(fig_row, fig_col, i+1, xlabel=str(i)+' '+s)

        ca = Cell1D(rule, n)
        ca.start_string(s)
        ca.loop(n-1)
        ca.draw()

    fig.tight_layout()
    return fig

def draw_seeded_ca(rule, val, n=1000):

    fig = plt.figure(figsize=(16,10))
    s = '{0:b}'.format(val)
    ca = Cell1D(rule, n)
    ca.start_string(s)
    ca.loop(n-1)
    ca.draw()

    fig.tight_layout()
    return fig

def draw_tm(table, n=32):
    """Makes and draws a Turing machine with a given rule table.

    table: dictionary of TM rules
    n: number of rows
    returns: matplotlib.pyplot.figure
    """
    tm = Turing(table, n)
    tm.loop(n-1)
    fig = tm.draw()
    return fig

def lcg(m, a, c):
    """Implements a linear congruential generator for random number generation.
    """
    seed = time.process_time_ns()

    # Check pre-conditions on inputs
    if (m<=0) or (not isinstance(m, int)): raise ValueError("'m' modulus must be positive integer")
    if (a<=0) or (a>=m) or (not isinstance(a, int)): raise ValueError("'a' must be a postive \
            integer satisfying 0 < a < m")
    if (c < 0) or (c >= m) or (not isinstance(c, int)): raise ValueError("'c' must be a positive \
            integer satisfying 0 <= c < m")
    if (seed < 0) or (seed >= m) or (not isinstance(seed, int)): raise ValueError("'seed' must be \
            a positive integer satisfying 0 <= seed < m")

    while True:
        seed = (a * seed + c) % m
        yield bin(seed)

def life():
    a = np.random.randint(2, size=(10, 10), dtype=np.uint8)
    b = np.zeros_like(a)
    rows, cols = a.shape

    kernel = np.array([ [1,1,1],
                        [1,10,1],
                        [1,1,1]])
    c = correlate2d(a, kernel, mode='same')
    b = (c==3) | (c == 12) | (c == 13)
    b = b.astype(np.uint8)

if __name__=='__main__':
    """Runs various functions. Split into cells for cell-wise execution
    if applicable.
    """
    ## Test correlation
    #N = 10
    #row = np.arange(N, dtype=np.uint8)
    #window = [1,1]
    #c = my_correlate(row, window, mode='same')
    #print(c)

    ## Run Rule 110 CA

    #n = 100
    #bits = 6
    #fig = iterate_ca_grid(110, n=n, bits=bits)
    #plt.show(block=True)

    ## Run Rule 110 CA for specific starting strings

    #n = 500
    #val = 365
    #rule = 110
    #fig = draw_seeded_ca(rule, val, n)
    #plt.show(block=True)
    #iterate_ca(rule, n=n)

    ## Create and draw 3-state busy-beaver Turing machine
    #table ={}
    ##table[0/1, 'A/B/C'] = (0/1, 'L/R', 'A/B/C')
    #table[0, 'A'] = (1 ,'L', 'C')
    #table[0, 'B'] = (1 ,'R', 'A')
    #table[0, 'C'] = (1 ,'R', 'B')
    #table[1, 'A'] = (0 ,'R', 'B')
    #table[1, 'B'] = (1 ,'R', 'C')
    #table[1, 'C'] = (0 ,'L', 'A')

    #n=100
    #fig = draw_tm(table, n)
    #plt.show(block=True)

    ## Test LCG RNG
    #m, a, c = (2**32, 1103515245, 12345)
    #gen = lcg(m, a, c)
    #for i in range(10):
    #    out = next(gen)
    #    out = int(out, 32)
    #    out = str(out)
    #    sys.stdout.write(out)
