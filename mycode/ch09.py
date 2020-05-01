## Import modules and boilerplate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import decorate, savefig, underride, three_frame, set_palette
from scipy.signal import correlate2d
from Cell2D import Cell2D, draw_array
from matplotlib.colors import LinearSegmentedColormap

# make a custom color map
palette = sns.color_palette('muted')
colors = 'white', palette[1], palette[0]
cmap = LinearSegmentedColormap.from_list('cmap', colors)

## Class and function definitions
class Schelling(Cell2D):
    """Represents a grid of Schelling agents."""

    options = dict(mode='same', boundary='wrap')

    kernel = np.array([ [1,1,1],
                        [1,0,1],
                        [1,1,1]], dtype = np.int8)

    def __init__(self, m, p):
        """Initializes Schelling model.
        m: rows and columns of array
        p: proportion of neighbors required for happy agent"""

        self.p = p
        # 0 is empty, 1 is red, 2 is blue
        choices = np.array([0, 1, 2], dtype=np.int8)
        probs = [0.1, 0.45, 0.45]
        self.array = np.random.choice(choices, (m, m), p=probs)

    def count_neighbors(self):
        """Survey neighboring cells.

        return: tuple of
            empty: True where cells are empty
            frac_red: fraction of red neighbors around each cell
            frac_blue: fraction of blue neighbors around each cell
            frac_same: fraction of neighbors with the same color

        """
        a = self.array

        empty = a==0
        red = a==1
        blue = a==2

        # count red neighbors, blue neighbors, and total
        num_red = correlate2d(red, self.kernel, **self.options)
        num_blue = correlate2d(blue, self.kernel, **self.options)
        num_neighbors = num_red + num_blue

        # compute fraction of similar neighbors
        frac_red = num_red / num_neighbors
        frac_blue = num_blue / num_neighbors

        # no neighbors is considered the same as no similar neighbors
        # this is an arbitrary choice for a rare event
        frac_red[num_neighbors == 0] = 0
        frac_blue[num_neighbors == 0] = 0

        # for each cell, compute the fraction of neighbors of same color
        frac_same = np.where(red, frac_red, frac_blue)

        # for empty cells, frac_same is NaN
        frac_same[empty] = np.nan

        return empty, frac_red, frac_blue, frac_same

    def segregation(self):
        """Computes the average fraction of similar neighbors.

        returns: fraction of similar neighbors, averaged over cells
        """
        _, _, _, frac_same = self.count_neighbors()
        return np.nanmean(frac_same)

    def step(self):
        """Executes one time step for Schelling object.

        returns: fraction of similar neighbors, averaged over cells
        """
        a = self.array
        empty, _, _, frac_same = self.count_neighbors()

        # find the unhappy cells (ignore NaN in frac_same)
        with np.errstate(invalid='ignore'):
            unhappy = frac_same < self.p
        unhappy_locs = locs_where(unhappy)

        # find the empty cells
        empty_locs = locs_where(empty)

        # find the empty cells
        if len(unhappy_locs):
            np.random.shuffle(unhappy_locs)

        # for each unhappy cell, choose a random destination
        num_empty = np.sum(empty)
        for source in unhappy_locs:
            i = np.random.randint(num_empty)
            dest = empty_locs[i]

            # move
            a[dest] = a[source]
            a[source] = 0
            empty_locs[i] = source

        # check that the number of empty cells is unchanged
        num_empty2 = np.sum(a==0)
        assert num_empty == num_empty2

        # return the average fraction of similar neighbors
        return np.nanmean(frac_same)

    def draw(self, **options):
        """Draw Schelling model."""
        return Cell2D.draw(self, cmap=cmap, vmax=2, origin='lower', **options)

    def animate(self, **options):
        """Animate Schelling model."""
        return Cell2D.animate(self, cmap=cmap, vmax=2, **options)

class Bishop(Schelling):
    """Implements a segregation model that moves 'k' neighbors each step
    and moves them to the most similar empty space available."""

    def __init__(self, m, k):
        """Initializes Schelling model.
        m: rows and columns of array
        p: proportion of neighbors required for happy agent"""

        self.k = k
        # 0 is empty, 1 is red, 2 is blue
        choices = np.array([0, 1, 2], dtype=np.int8)
        probs = [0.1, 0.45, 0.45]
        self.array = np.random.choice(choices, (m, m), p=probs)

    def step(self):
        """Executes one time step for Bishop object.

        returns: fraction of similar neighbors, averaged over cells
        """
        a = self.array
        empty, frac_red, frac_blue, frac_same = self.count_neighbors()

        # find the empty cells
        empty_locs = locs_where(empty)

        # select 'k' non-empty cells
        filled_locs = locs_where(~empty)
        ind = np.random.choice(len(filled_locs), self.k, replace=False)
        moving_locs = [filled_locs[i] for i in ind]

        # for each unhappy cell, choose a random destination
        num_empty = np.sum(empty)

        # make list of frac of red and blue neighbors for empty cells

        for source in moving_locs:

            # calculate the neighborhoods of empty cells
            empty, frac_red, frac_blue, frac_same = self.count_neighbors()
            empty_red = [frac_red[loc] for loc in empty_locs]
            empty_blue = [frac_blue[loc] for loc in empty_locs]

            # find empty destination with most same neighborhood
            empty_same = empty_red if a[source] == 1 else empty_blue
            i = np.argmax(empty_same)
            dest = empty_locs[i]

            # move
            a[dest] = a[source]
            a[source] = 0
            empty_locs[i] = source
            empty_same[i] = frac_same[source]

        # check that the number of empty cells is unchanged
        num_empty2 = np.sum(a==0)
        assert num_empty == num_empty2

        # return the average fraction of similar neighbors
        return np.nanmean(frac_same)

def locs_where(condition):
    """Find cells where a logical array is True.

    condition: logical array

    returns: list of location tuples
    """
    return list(zip(*np.nonzero(condition)))

def seg_v_steps():
    """Plot segregation vs time steps for different p values.
    """
    np.random.seed(17)
    set_palette('Blues', 5, reverse=True)

    print('p  Final segregation value  Final seg val - p')
    for p in [0.5, 0.4, 0.3, 0.2]:
        grid = Schelling(100, p)
        segs = [grid.step() for i in range(12)]
        plt.plot(segs, label='p = %.1f' %p)
        print(p, segs[-1], segs[-1] - p)

    decorate(xlabel='Time Steps', ylabel='Segregation',
            loc='lower right', ylim=[0, 1])

    savefig('myfigs/chap09-2')

    plt.show()

def seg_v_steps_bishop(k):
    """Plot segregation vs k."""
    np.random.seed(17)

    grid = Bishop(100, 1)
    res = []
    for s in range(k):
        res.append(grid.step())

    plt.plot(res, label='Segregation')
    decorate(xlabel='Moves', ylabel='Segregation')

    savefig('myfigs/chap09-5')
    plt.show(block=True)

if __name__ == '__main__':
    """Scripts for execution of Schelling model."""

    ## Scripts below here
    seg_v_steps_bishop(1000)
