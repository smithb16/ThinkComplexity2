from Cell2D import Cell2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import underride

class Turmite(Cell2D):

    directions = ['N','E','S','W']
    # compass: N,     E,     S,       W
    compass = [(1,0), (0,1), (-1, 0), (0, -1)]
    rules = {0:1, 1: -1}

    def __init__(self, m, n=None):
        """Initialize Turmite.
        Currently only built for Langston's Ant.
        m: number of rows
        n: number of columns
        """
        Cell2D.__init__(self, m, n)
        self.direction = 0 #Initialize pointing North
        #self.location syntax: (row, col)
        self.location = (m//2, n//2)

    def step(self):
        """Move ant through one step. Currently only works for
        Langstons Ant.
        """
        row, col = self.location
        self.turn()
        self.array[row, col] = (self.array[row, col] + 1) % 2
        self.forward()

    def turn(self):
        """Turn ant basted on rules."""
        val = self.array[self.location]
        rotate = Turmite.rules[val]
        self.direction = (self.direction + rotate) % 4

    def forward(self):
        """Move ant forward one unit."""
        row, col = self.location
        d_row, d_col = Turmite.compass[self.direction]
        self.location = (row + d_row, col + d_col)

    def init_fig(self, **options):
        """Initialize figure for animation and drawing."""
        options = underride(options,
                            cmap='Greens',
                            alpha=0.7,
                            vmin=0, vmax=1,
                            interpolation='none',
                            origin='lower',
                            extent=[0, self.n, 0, self.m])

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.array, **options)
        row, col = self.location
        self.line, = self.ax.plot(col+0.5, row+0.5, 'r.')

    def update_anim(frame, self):
        """Update frame of animation"""
        self.step()
        self.im.set_data(self.array)
        row, col = self.location
        self.line.set_data(col+0.5, row+0.5)

    def draw(self):
        """Draw array and ant."""
        self.init_fig()
        self.im.set_data(self.array)
        row, col = self.location
        self.ax.plot(col+0.5, row+0.5, 'r.')
        return self.fig

if __name__ == '__main__':

    ant = Turmite(100, 100)
    ani = ant.animate()
    plt.show(block=True)
