## Module import and boilerplate
import matplotlib.pyplot as plt
import numpy as np
import itertools

from Cell2D import Cell2D, draw_array
from empiricaldist import Pmf, Cdf
from utils import decorate, savefig, underride
from scipy.signal import correlate2d, welch
from scipy.stats import linregress

## Class and function definitions
class SandPile(Cell2D):
    """Represents a sand pile per Bak, Tang, Wiesenfeld."""

    kernel = np.array([ [0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]])

    def __init__(self, m, n=None, level=9, source='even'):
        """Initialize sand pile.
        m: rows in array
        n: columns in array
        level: height of each cell at start
        source: 'even' to initialize each cell with level
                'single' to initialize center cell with level * m * n and all others to zero
        """
        self.m = m
        self.n = m if n is None else n
        n = self.n
        if source == 'even':
            self.array = np.ones((m, n), dtype=np.int32) * level
        elif source == 'single':
            self.array = np.zeros((m, n), dtype=np.int32)
            self.array[m//2, n//2] = level * m * n
        self.toppled_seq = []

    def step(self, k=3):
        """Executes one time step.

        returns: number of cells that toppled
        """
        toppling = self.array > k
        num_toppled = np.sum(toppling)
        self.toppled_seq.append(num_toppled)

        c = correlate2d(toppling, self.kernel, mode='same')
        self.array += c
        return num_toppled

    def drop(self, iters=1):
        """Adds one increment to a random cell."""
        a = self.array
        m, n = a.shape
        for i in range(iters):
            index = np.random.randint(m), np.random.randint(n)
            a[index] += 1

    def run(self):
        """Runs until equilibrium.
        returns: duration, total_toppled
        """
        total=0
        for i in itertools.count(1):
            num_toppled = self.step()
            total += num_toppled
            if num_toppled == 0:
                return i, total

    def drop_and_run(self):
        """Drops a random grain and runs until equilibrium.
        returns: duration, total_toppled
        """
        self.drop()
        duration, total_toppled = self.run()
        return duration, total_toppled

    def draw(self, **options):
        options = underride(options,
                            cmap='YlOrRd', vmax=5)
        self.fig = Cell2D.draw(self, **options)
        return self.fig

    def animate(self, **options):
        options = underride(options,
                            cmap='YlOrRd', vmax=5)
        ani = Cell2D.animate(self, **options)
        return ani

def slope(xs, ys):
    return np.diff(np.log(ys)) / np.diff(np.log(xs))

def draw_four(pile, levels=range(4)):
    plt.figure(figsize=(8,8))
    for i, level in enumerate(levels):
        plt.subplot(2, 2, i+1)
        draw_array(pile.array==level, cmap='YlOrRd', vmax=1)

    plt.tight_layout()

def count_cells(a):
    """Counts the number of cells in a box of increasing size.
    a: boolean array

    returns: list of (i, i**2, cell count) tuples
    """
    m, n = a.shape
    end = min(m, n)

    res = []
    for i in range(1,end, 2):
        top = (m-i)//2
        left = (n-i)//2
        box = a[top:top+i, left:left+i]
        total = np.sum(box)
        res.append((i, i**2, total))

    return np.transpose(res)

def box_count(pile, level, plot=False):
    """Estimates the fractal dimension by box counting.

    pile: SandPile
    level: which level from the pile to count
    plot: boolean, whether to generate plot

    returns: estimated fractal dimension
    """
    res = count_cells(pile.array==level)
    steps, steps2, cells = res

    # select the range where we have a nonzero number of cells
    legit = np.nonzero(cells)
    steps = steps[legit]
    steps2 = steps2[legit]
    cells = cells[legit]

    if plot:
        # only put labels on the left and bottom subplots
        xlabel = 'Box Size' if level in [2, 3] else ''
        ylabel = 'Cell Count' if level in [0, 2] else ''

        options = dict(linestyle='dashed', color='gray', alpha=0.7)
        plt.plot(steps, steps2, **options)
        plt.plot(steps, cells, label='level=%d' % level)
        plt.plot(steps, steps, **options)

        decorate(xscale='log', yscale='log',
                xlim=[1, 200], loc='upper left',
                xlabel=xlabel, ylabel=ylabel)

    params = linregress(np.log(steps), np.log(cells))
    print('Slope: ',params[0])
    return params[0]

def box_count_four(pile, levels=range(4)):
    """Applies box counting to each level in the pile.

    pile: SandPile
    levels: list of levels to check
    """
    plt.figure(figsize=(8,8))

    dims = []
    for i, level in enumerate(levels):
        plt.subplot(2, 2, i+1)
        dim = box_count(pile, level, plot=True)
        dims.append(dim)

    return dims

if __name__ == '__main__':
    """Execute scripts."""

    ### Generate sand pile and test many avalanches
    #pile2 = SandPile(m=50, level=30)
    #pile2.run()
    #iters=100000
    #res = [pile2.drop_and_run() for _ in range(iters)]

    #T, S = np.transpose(res)
    #T = T[T>1]
    #S = S[S>0]

    ### Make and plot PMFs
    #pmfT = Pmf.from_seq(T)
    #pmfS = Pmf.from_seq(S)

    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)

    #pmfT.plot(label='T')
    #decorate(xlabel='Avalanche duration',
    #                 ylabel='PMF',
    #                 xlim=[1, 50], loc='upper right')

    #plt.subplot(1, 2, 2)
    #pmfS.plot(label='S')
    #decorate(xlabel='Avalanche size',
    #                 xlim=[1, 50])

    ### Make PMF & CDF  on log-log
    #plt.figure(figsize=(10, 10))

    ## Subplot1 - PMF of T
    #plt.subplot(2, 2, 1)

    #xs = [2, 600]
    #ys = [1.3e-1, 2.2e-4]
    #print(slope(xs, ys))

    #options = dict(color='gray', alpha=0.3, linewidth=4)
    #plt.plot(xs, ys, **options)

    #pmfT.plot(label='T', linewidth=2)
    #decorate(xlabel='Avalanche duration',
    #                 xlim=[1, 1000],
    #                 ylabel='PMF',
    #                 xscale='log',
    #                 yscale='log',
    #                 loc='upper right')

    ## Subplot 2 - PMF of S
    #plt.subplot(2, 2, 2)

    #xs = [1, 5000]
    #ys = [1.3e-1, 2.3e-5]
    #print(slope(xs, ys))

    #plt.plot(xs, ys, **options)
    #pmfS.plot(label='S', linewidth=1)
    #decorate(xlabel='Avalanche size',
    #                 xlim=[1, 5600],
    #                 xscale='log',
    #                 yscale='log')

    ## Make CDFs
    #cdfS = Cdf.from_seq(S)
    #cdfT = Cdf.from_seq(T)

    ## Subplot 3 - CDF T
    #plt.subplot(2,2,3)
    #(1-cdfT).plot(color='C0', label='T')
    #decorate(xlabel='Avalanche Size',xscale='log',
    #        ylabel='CCDF', yscale='log')

    ## Subplot 4 - CDF S
    #plt.subplot(2,2,4)
    #(1-cdfS).plot(color='C0', label='S')
    #decorate(xlabel='Time Steps',xscale='log',
    #        ylabel='CCDF', yscale='log')


    #savefig('myfigs/chap08-7')
    #plt.show(block=True)

    ### Look for fractals
    #pile3 = SandPile(m=160, level=22)
    #pile3.run()
    #fig = pile3.draw()
    #plt.show(block=True)

    ### Draw the sand pile at different levels
    #draw_four(pile3)
    #savefig('myfigs/chap08-4')
    #plt.show(block=True)

    ### Draw the individual levels of pile3
    #dims = box_count_four(pile3)
    #print(dims)
    #savefig('myfigs/chap08-5')
    #plt.show(block=True)

    ### Analyze spectral power density of toppling sequence
    #signal = pile2.toppled_seq
    #nperseg = 10_000
    #freqs, powers = welch(signal, nperseg=nperseg, fs = nperseg)

    #x = nperseg
    #ys = np.array([x**1.58, 1]) / 2.7e3
    #plt.plot([1, x], ys, color='gray', linewidth=1)

    #plt.plot(freqs, powers)
    #decorate(xlabel='Frequency',
    #        xscale='log',
    #        xlim=[1, 1200],
    #        ylabel='Power',
    #        yscale='log',
    #        ylim=[1e-4, 5])

    #savefig('myfigs/chap08-6')
    #plt.show()

    ### Make graph for fractal analysis
    #pile5 = SandPile(m=160, level=20)
    #pile5.run()
    #iters=10000
    #res = [pile5.drop_and_run() for _ in range(iters)]

    ### Perform fractal analysis for sand pile after random drops
    #draw_four(pile5)
    #savefig('myfigs/chap08-8')
    #plt.show(block=True)

    ### Draw the individual levels of pile3
    #dims = box_count_four(pile5)
    #print(dims)
    #savefig('myfigs/chap08-9')
    #plt.show(block=True)

    ### Perform fractal analysis of single-sourced pile.
    #pile6 = SandPile(m=80, level=8, source='single')
    #pile6.run()

    ### Perform fractal analysis for sand pile after random drops
    #draw_four(pile6)
    #savefig('myfigs/chap08-10')
    #plt.show(block=True)

    ## Draw the individual levels of pile3
    #dims = box_count_four(pile6)
    #print(dims)
    #savefig('myfigs/chap08-11')
    #plt.show(block=True)

