## Boilerplate and modules

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.signal import correlate2d
from Cell2D import Cell2D
from utils import three_frame, decorate, savefig, underride
from matplotlib.animation import FuncAnimation

class ForestFire(Cell2D):

    kernel = np.array([ [0,1,0],
                        [1,5,1],
                        [0,1,0]])

    def __init__(self, m, p, f):
        """Instantiate ForestFire.
        m: number of rows and columns in array
        p: probability of empty cell growing new tree
        f: probability of cell with tree spontaneously combusting
        """
        Cell2D.__init__(self, m)
        self.p = p
        self.f = f
        self.start_random(0.4)

    def step(self):
        """ Progress forest fire by one time step.
        """
        m, n = self.array.shape
        p, f = self.p, self.f
        a = self.array
        new_trees = np.random.choice([1,0], size=(m, n), p=[p, 1-p])
        new_fires = np.random.choice([1,0], size=(m, n), p=[f, 1-f])
        c = correlate2d(a, self.kernel, mode='same')

        #print('a:\n', a)
        #print('c:\n', c)
        #print('new_trees:\n', new_trees)
        #print('new_fires:\n', new_fires)

        # Trees on fire
        self.array[(c<25) & (c>=10) & (a==1)] = 5

        # Trees burned down
        self.array[c>=25] = 0

        # Add new trees
        self.array[(a == 0) & (new_trees == 1)] = 1

        # Add new fires
        self.array[(a == 1) & (new_fires == 1)] = 5

    def num_trees(self):
        return np.sum(self.array == 1)

    def num_burning(self):
        return np.sum(self.array == 5)

def test_fire(fire, iters=400, window=100):
    """Steps ForestFire object until number of trees stabilizes.
    fire: ForestFire object
    """
    size = fire.m
    fire.loop(iters)

    trees = []
    burning = []
    for i in range(window):
        t, b = fire.num_trees(), fire.num_burning()
        trees.append(t)
        burning.append(b)

    return np.mean(trees), np.mean(burning)

def plot_fire_scaling(p, f, sizes=np.logspace(1.5,3,6,dtype=int)):
    """Count the number of trees and number of burning trees in stable forest fire.
    p: proportion of trees added each step
    f: proportion of new fires added each step
    sizes: iterable of array sizes
    """
    res = []
    for size in sizes:
        fire = ForestFire(size, p, f)
        n_trees, n_burning = test_fire(fire)
        res.append((size, size**2, n_trees, n_burning))

    sizes, cells, trees, burning = zip(*res)

    options = dict(linestyle='dashed', color='gray', alpha=0.7)

    fig, ax = plt.subplots()
    ax.plot(sizes, cells, label='d=2', **options)
    ax.plot(sizes, trees, 'k.', label='Trees')
    ax.plot(sizes, burning, 'r.', label='Burning')
    ax.plot(sizes, sizes, label='d=1', **options)

    decorate(   xlabel = 'Array Size',
                ylabel = 'Cell Count',
                xscale = 'log', xlim = [10, 2000],
                yscale = 'log', ylim = [3, 2000000],
                loc = 'upper left')

    for ys in [cells, trees, burning, sizes]:
        params = linregress(np.log(sizes), np.log(ys))
        print('Slope of lines:\n', params[0])

    return fig

def trees_vs_time(m, p, f, iters):
    fire = ForestFire(m, p, f)
    res = []

    for step in range(iters):
        res.append((step, 40*fire.num_burning(), fire.num_trees()))
        fire.step()

    steps, burning, trees = zip(*res)

    fig, ax = plt.subplots()
    ax.plot(steps, burning, label='40 x Burning')
    ax.plot(steps, trees, label='Trees')
    decorate(   xlabel = 'Step',
                xlim = [0, iters],
                ylim = [0, 0.5 * m**2],
                loc = 'upper left')

    return fig

if __name__ == '__main__':
    p, f = 0.01, 0.00001
    #fire = ForestFire(200, p, f)
    #fig = trees_vs_time(100, p, f, 1000)
    fig = plot_fire_scaling(p, f)

    options = dict(vmax=3)
    #ani = fire.animate(**options)
    savefig('myfigs/fire_scaling_p01f00001.png')
    plt.show(block=True)
