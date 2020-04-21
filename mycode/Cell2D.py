import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from utils import underride

class Cell2D:

    def __init__(self, m, n=None):

        self.m = m
        self.n = m if n is None else n
        self.array = np.zeros((self.m,self.n), np.uint8)

    def init_fig(self, **options):
        options = underride(options,
                            cmap='Greens',
                            alpha=0.7,
                            vmin=0, vmax=1,
                            interpolation='none',
                            origin='upper',
                            extent=[0, self.n, 0, self.m])

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.array, **options)

    def add_cells(self, row, col, *strings):
        """Adds cells at the given location.
        row: top row index
        col: left col index
        strings: list of strings of 0s and 1s
        """
        for i, s in enumerate(strings):
            self.array[row+i, col:col+len(s)] = np.array([int(b) for b in s])


    def loop(self, iters=1):
        """Runs the given number of steps."""
        for i in range(iters):
            self.step()

    def draw(self):
        self.init_fig()
        self.im.set_data(self.array)
        return self.fig

    def step(self):
        m, n = self.array.shape
        x = np.random.randint(0, m-1)
        y = np.random.randint(0, n-1)
        self.array[y,x] = (self.array[y,x] + 1) % 2

    def animate(self):
        self.init_fig()
        ani = FuncAnimation(self.fig, self.__class__.update_anim, fargs=(self,),
                interval=50)
        return ani

    def update_anim(frame, self):
        """Update frame of animation"""
        self.step()
        self.im.set_data(self.array)

if __name__ == '__main__':
    life = Cell2D(10)
    ani = life.animate()
    #life.step()
    #fig = life.draw()
    plt.show(block=True)
