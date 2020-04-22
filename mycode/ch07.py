## Boilerplate and modules

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Cell1D import Cell1D, draw_ca
from scipy.stats import linregress
from scipy.signal import correlate2d
from Cell2D import Cell2D
from utils import three_frame, decorate, savefig, underride
from matplotlib.animation import FuncAnimation

## Start class and function definitions here

class Diffusion(Cell2D):

    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])

    def __init__(self, m, r=0.01):
        """Initialize square diffusion.
        m: number of rows
        r: rate of diffusion
        """
        self.array = np.zeros((m,m), dtype=float)
        self.r = r

    def step(self):
        """Execute one time step"""
        kernel = self.__class__.kernel
        c = correlate2d(self.array, kernel, mode='same')
        self.array += self.r * c

    def draw(self):
        op1 = dict(cmap='Reds')
        op2 = dict(cmap='Blues')
        self.init_fig(**op1)
        self.im.set_data(self.array)
        return self.fig



class ReactionDiffusion(Cell2D):
    """Represents 2-chemical diffusion reactions."""

    kernel = np.array([ [0.05, 0.2, 0.05],
                        [0.2, -1.0, 0.2],
                        [0.05, 0.2, 0.05]])

    def __init__(self, m, params, noise=0.1):
        """Initialize 2-chemical diffusion reaction.
        m: int number of rows
        params: tuple of parameters for reaction
            syntax: (ra, rb, f, k)
                    (diffusion rate of a, diffusion rate of b,
                    feed rate of a, kill rate of b)
        noise: float value of random noise of chemical 2
        """
        self.params = params
        self.array = np.ones((m, m), dtype = float)
        self.array2 = noise * np.random.random((m,m))
        add_island(self.array2)

    def step(self):
        """Step for ReactionDiffusion."""
        A = self.array
        B = self.array2
        ra, rb, f, k = self.params

        options = dict(mode='same', boundary='wrap')

        # Apply diffusion kernel to A & B
        cA = correlate2d(A, self.kernel, **options)
        cB = correlate2d(B, self.kernel, **options)

        # Reaction of A and B defined by:
        reaction = A * B**2

        # Assume reaction consumes A and produces B
        self.array += ra * cA - reaction + f * (1-A)
        self.array2 += rb * cB + reaction - (f+k) * B

    def draw(self):
        #Set options for both images
        options = dict(interpolation='bicubic',
                        vmin=None, vmax=None,
                        cmap='Reds',
                        origin='lower')

        # Set specific options for im2
        options2 = underride(dict(cmap='Blues'), **options)

        # Initialize figure
        self.init_fig(**options)

        # Add a second imshow to axes
        self.im2 = self.ax.imshow(self.array2, **options2)

        # Set data for both images
        self.im.set_data(self.array)
        self.im2.set_data(self.array2)

        return self.fig

    def update_anim(frame, self):
        """Update frame of animation"""
        self.step()
        self.im.set_data(self.array)
        self.im2.set_data(self.array2)

    def animate(self):

        # Set options for both images
        options = dict(interpolation='bicubic',
                        vmin=None, vmax=None,
                        cmap='Reds',
                        origin='lower')

        # Set specific options for im2
        options2 = underride(dict(cmap='Blues'), **options)

        # Initialize fig and make im
        self.init_fig(**options)

        # Make im2
        self.im2 = self.ax.imshow(self.array2, **options2)

        ani = FuncAnimation(self.fig, self.__class__.update_anim, fargs=(self,),
                interval=20)
        return ani

class Percolation(Cell2D):
    """Represents percolation through a porous medium"""

    kernel = np.array([ [0,1,0],
                        [1,0,1],
                        [0,1,0]])

    def __init__(self, m, q):
        """Initialize percolation.
        m: number of rows and columns
        q: probability of cell being porous
        """
        self.q = q
        self.array = np.random.choice([1,0], (m,m), p=[q, 1-q])
        self.array[0] = 5

    def step(self):
        """Step percolation forward"""
        a = self.array
        c = correlate2d(a, self.kernel, mode='same')
        self.array[(a == 1) & (c>=5)] = 5

    def num_wet(self):
        """Return number of wet cells"""
        return np.sum(self.array == 5)

    def bottom_row_wet(self):
        """Return number of wet cells in the bottom row."""
        return np.sum(self.array[-1] == 5)

    def draw(self):
        """Draw method for percolation."""
        options = dict(vmax = 5)
        self.init_fig(**options)
        self.im.set_data(self.array)
        return self.fig

    def animate(self):
        """Animate method for percolation."""
        options = dict(vmax = 5)
        ani = Cell2D.animate(self, **options)
        return ani

def test_perc(perc):
    """Test whether fluid has percolated to bottom cell.
    perc: Percolation object
    """
    num_wet = perc.num_wet()

    while True:
        perc.step()

        if perc.bottom_row_wet():
            return True

        new_num_wet = perc.num_wet()
        if new_num_wet == num_wet:
            return False

        num_wet = new_num_wet

def plot_rd(f, k, n=5000):
    """Makes a ReactionDiffusion object with given parameters.
    """
    params = 0.5, 0.25, f, k
    rd = ReactionDiffusion(100, params)
    rd.loop(n)
    fig = rd.draw()
    plt.show()

def add_island(a, height=0.1):
    """Add island of height to array.
    a: np.array
    height: float of height
    """
    m, n = a.shape
    radius = min(m, n)//20
    i = m//2
    j = n//2
    a[i-radius:i+radius, j-radius:j+radius] += height

def estimate_prob_percolating(m=100, q=0.5, iters=100):
    """Find probability of percolating using iterative approach.
    m: size of array
    q: probability of porous cell
    iters: number of iterations to run
    """
    t = [test_perc(Percolation(m, q)) for i in range(iters)]
    return np.mean(t)

def find_critical(m=100, q=0.6, iters=100):
    qs = [q]
    for i in range(iters):
        perc = Percolation(m, q)
        if test_perc(perc):
            q -= 0.005
        else:
            q += 0.005
        qs.append(q)
    return qs

def count_cells(rule, n=500):
    """Count cells in 1-D CA.
    rule: int of rule for Wolfram's CA
    n: number of rows
    """
    ca = Cell1D(rule, n)
    ca.start_single()

    res = []
    for i in range(1, n):
        cells = np.sum(ca.array)
        res.append((i, i**2, cells))
        ca.step()

    return res

def test_fractal(rule, ax=None, plotting=False, ylabel='Number of Cells'):
    """Compute the fractal dimension of a rule.
    rule: int rule for Wolfram's 1-D CA
    ax: matplotlib.Axes
    ylabel: string
    returns: ax: matplotlib.Axes
    """
    res = count_cells(rule)
    steps, steps2, cells = zip(*res)

    for ys in [cells]:
        params = linregress(np.log(steps), np.log(ys))
        print('Slope of Rule %i: ' % rule, params[0])

    if plotting:
        if ax == None:
            fig, ax = plt.subplots()

        options = dict(linestyle='dashed', color='gray', alpha=0.7)
        ax.plot(steps, steps2, label='d=2')
        ax.plot(steps, cells, label='Rule %i' % rule)
        ax.plot(steps, steps, label='d=1')

        decorate( xscale='log', yscale='log',
                xlabel='Time Steps', ylabel=ylabel,
                xlim = [1,600], loc='upper left')

    return ax, params[0], rule

def cycle_fractals():

    for rule in range(256):
        _, slope, rule = test_fractal(rule)

def plot_perc_scaling(q, sizes=np.logspace(1,2,50,dtype=int)):
    """Count the number of cells filled across range of sizes.
    q: proportion of porous cells
    sizes: iterable of array sizes
    """
    res = []
    for size in sizes:
        perc = Percolation(size, q)
        if test_perc(perc):
            num_filled = perc.num_wet() - size
            res.append((size, size**2, num_filled))

    sizes, cells, filled = zip(*res)

    options = dict(linestyle='dashed', color='gray', alpha=0.7)

    fig, ax = plt.subplots()
    ax.plot(sizes, cells, label='d=2', **options)
    ax.plot(sizes, filled, 'k.', label='filled')
    ax.plot(sizes, sizes, label='d=1', **options)

    decorate(   xlabel = 'Array Size',
                ylabel = 'Cell Count',
                xscale = 'log', xlim = [9, 110],
                yscale = 'log', ylim = [9, 20000],
                loc = 'upper left')
    plt.show()

    for ys in [cells, filled, sizes]:
        params = linregress(np.log(sizes), np.log(ys))
        print('Slope of lines:\n', params[0])


if __name__ == '__main__':

    cycle_fractals()
    plt.show()
    #plot_perc_scaling(0.59)

    #qs = find_critical()
    #print('qs: ', qs)
    #print(np.mean(qs))
    #params1 = 0.5, 0.25, 0.035, 0.057   # pink spots and stripes
    #params2 = 0.5, 0.25, 0.055, 0.062   # coral
    #params3 = 0.5, 0.25, 0.039, 0.065   # blue spots

    #plot_rd(0.039, 0.065)

    #params = params3
    #diff = ReactionDiffusion(100, params)
    #ani = diff.animate()
    #fig = diff.draw()
    #plt.show()


    #react = ReactionDiffusion(300, 300, params)
