## Import modules and boilerplate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from empiricaldist import Pmf, Cdf
from utils import decorate, savefig, underride, three_frame, set_palette
from scipy.signal import correlate2d
from Cell2D import Cell2D, draw_array
from matplotlib.colors import LinearSegmentedColormap

## Class and function definition here

def make_locs(m, n):
    """Make array where each row is an index in an 'm' by 'n' grid.

    m: int number of rows
    n: int number of cols

    return: Numpy array
    """
    t = [(i, j) for i in range(m) for j in range(n)]
    return np.array(t)

def make_visible_locs(vision):
    """Computes the kernel of visible cells.

    vision: int distance each agent can see
    """
    def make_array(d):
        """Generate visible cells with increasing distance."""
        a = np.array([[-d, 0], [d, 0], [0, -d], [0, d]])
        np.random.shuffle(a)
        return a

    arrays = [make_array(d) for d in range(1, vision+1)]
    return np.vstack(arrays)

def distances_from(m, i, j):
    """Computes an array of distances.

    m: size of array
    i, j: coordinate to find distance from

    returns: array of float
    """
    X, Y = np.indices((m, m))
    return np.hypot(X-i, Y-j)

class Sugarscape(Cell2D):
    """Represent an Epstein-Axtell Sugarscape."""

    def __init__(self, m, **params):
        """Initialize the attributes.

        m: number of rows and columns
        params: dictionary of parameters
        """
        self.m = m
        self.params = params

        # track variables
        self.agent_count_seq = []

        # make the capacity array
        self.capacity = self.make_capacity()

        # initially all cells are at capacity
        self.array = self.capacity.copy()

        # make the agents
        self.make_agents()

    def make_capacity(self):
        """Make capacity array."""

        # compute the distance of each cell from the peaks
        dist1 = distances_from(self.m, 15, 15)
        dist2 = distances_from(self.m, 35, 35)
        dist = np.minimum(dist1, dist2)

        # cells in the capacity array are set according to dist from peak
        bins = [21, 16, 11, 6]
        a = np.digitize(dist, bins)
        return a

    def make_agents(self):
        """Makes the agents."""

        # determine where the agents start and generate locations
        m, n = self.params.get('starting_box', self.array.shape)
        locs = make_locs(m, n)
        np.random.shuffle(locs)

        # make the agents
        num_agents = self.params.get('num_agents', 400)
        assert(num_agents <= len(locs))
        self.agents = [Agent(locs[i], self.params)
                        for i in range(num_agents)]

        # keep track of which cells are occupied
        self.occupied = set(agent.loc for agent in self.agents)

    def grow(self):
        """Adds sugar to all cells and caps them by capacity."""
        grow_rate = self.params.get('grow_rate', 1)
        self.array = np.minimum(self.array + grow_rate, self.capacity)

    def look_and_move(self, center, vision):
        """Finds the visible cell with the most sugar.

        center: tuple, coordinates of the center cell
        vision: int, maximum visible distance

        returns: tuple, coordinates of best cell
        """
        # find all visible cells
        locs = make_visible_locs(vision)
        locs = (locs + center) % self.m

        # convert rows of the array to tuples
        locs = [tuple(loc) for loc in locs]

        # select unoccupied cells
        empty_locs = [loc for loc in locs if loc not in self.occupied]

        # if all visible cells are occupied, stay put
        if len(empty_locs) == 0:
            return center

        # look up the sugar level in each empty visible cell
        t = [self.array[loc] for loc in empty_locs]

        # find the best one and return it
        # (in case of tie, argmax returns the first, which
        # is the closest)
        i = np.argmax(t)
        return empty_locs[i]

    def harvest(self, loc):
        """Removes and returns the sugar from 'loc'.

        loc: tuple coordinates
        """
        sugar = self.array[loc]
        self.array[loc] = 0
        return sugar

    def step(self):
        """Executes one time step."""
        replace = self.params.get('replace', False)

        # loop through the agents in random order
        random_order = np.random.permutation(self.agents)
        for agent in random_order:

            # mark the current cell unoccupied
            self.occupied.remove(agent.loc)

            # execute one step
            agent.step(self)

            # if the agent is dead, remove from the list
            if agent.is_starving() or agent.is_old():
                self.agents.remove(agent)
                if replace:
                    self.add_agent()

            else:
                # otherwise, mark its cell as occupied
                self.occupied.add(agent.loc)

        # update the time series
        self.agent_count_seq.append(len(self.agents))

        # regrow sugar
        self.grow()

        return len(self.agents)

    def add_agent(self):
        """Generates a new random agent.

        returns: new Agent
        """
        new_agent = Agent(self.random_loc(), self.params)
        self.agents.append(new_agent)
        self.occupied.add(new_agent.loc)
        return new_agent

    def random_loc(self):
        """Choose a random unoccupied cell.

        return: tuple coordinates
        """
        while True:
            loc = tuple(np.random.randint(self.m, size=2))
            if loc not in self.occupied:
                return loc

    def draw(self, **options):
        """Draw Sugarscape environment and agents."""
        options = underride(options,
                            cmap='YlOrRd',
                            vmax=9, origin='lower')
        self.init_fig(**options)
        self.im.set_data(self.array)
        return self.fig

    def init_fig(self, **options):
        """Initialize Sugarscape figure for animation and drawing."""
        options = underride(options,
                            cmap='YlOrRd',
                            vmax=9, origin='lower')
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.array, **options)

        # draw the agents
        xs, ys = self.get_coords()
        self.points = self.ax.plot(xs, ys, '.', color='red')[0]

    def update_anim(frame, self):
        """Update frame of animation"""
        self.step()
        self.im.set_data(self.array)

        # draw the agents
        xs, ys = self.get_coords()
        self.points.set_data(xs, ys)

    def get_coords(self):
        """Get the coordinates of the agents.

        Transforms from (row, col) into (x, y)

        returns: tuple of sequences, (xs, ys)
        """
        agents = self.agents
        ys, xs = np.transpose([agent.loc for agent in agents])
        return xs, ys

class Sugarscape_Grow(Sugarscape):
    """Represents an Epstein-Axtell Sugarscape with a new agent
    every step."""

    def __init__(self, m, **params):
        """Initialize the attributes.

        m: number of rows and columns
        params: dictionary of parameters
        """
        Sugarscape.__init__(self, m, **params)

        # track variables
        self.average_vision_seq = []
        self.average_metabolism_seq = []

    def step(self):
        """Executes one time step."""

        # add an agent
        self.add_agent()

        # measure population averages
        self.average_vision_seq.append(self.average_vision())
        self.average_metabolism_seq.append(self.average_metabolism())

        # take a Sugarscape step
        return Sugarscape.step(self)

    def average_vision(self):
        """Measure average vision of population."""
        return np.mean([agent.vision for agent in self.agents])

    def average_metabolism(self):
        """Measure average metabolism of population."""
        return np.mean([agent.metabolism for agent in self.agents])

class Agent:

    def __init__(self, loc, params):
        """Create a new agent at the given location.

        loc: tuple coordinates
        params: dictionary of parameters
        """
        self.loc = tuple(loc)
        self.age = 0

        # extract parameters
        max_vision = params.get('max_vision', 6)
        max_metabolism = params.get('max_metabolism', 4)
        min_lifespan = params.get('min_lifespan', 10000)
        max_lifespan = params.get('max_lifespan', 10000)
        min_sugar = params.get('min_sugar', 5)
        max_sugar = params.get('max_sugar', 25)

        # choose attributes
        self.vision = np.random.randint(1, max_vision + 1)
        self.metabolism = np.random.uniform(1, max_metabolism)
        self.lifespan = np.random.uniform(min_lifespan, max_lifespan)
        self.sugar = np.random.uniform(min_sugar, max_sugar)

    def step(self, env):
        """Look around, move, and harvest.

        env: Sugarscape
        """
        self.loc = env.look_and_move(self.loc, self.vision)
        self.sugar += env.harvest(self.loc) - self.metabolism
        self.age += 1

    def is_starving(self):
        """Check if sugar has gone negative."""
        return self.sugar < 0

    def is_old(self):
        """Check if lifespan is exceeded."""
        return self.age > self.lifespan

def vision_distribution(env):
    """Make CDF of vision distance.

    env: Sugarscape
    """
    cdf = Cdf.from_seq(agent.vision for agent in env.agents)
    cdf.plot()
    decorate(xlabel='Vision', ylabel='CDF')
    plt.show(block=True)

def metabolism_distribution(env):
    """Make CDF of metabolism distribution.

    env: Sugarscape
    """
    cdf = Cdf.from_seq(agent.metabolism for agent in env.agents)
    cdf.plot()
    decorate(xlabel='Metabolism', ylabel='CDF')
    plt.show(block=True)

def wealth_distribution(env, plot=True):
    """Make CDF of sugar distribution.

    env: Sugarscape
    """
    qs = [0.25, 0.5, 0.75, 0.9]
    cdf = Cdf.from_seq(agent.sugar for agent in env.agents)
    for q in qs:
        print('Wealth of {:.0%}'.format(q), end='')
        print(': %i' %cdf.quantile(q))

    if plot:
        cdf.plot()
        decorate(xlabel='Wealth', ylabel='CDF')
        plt.show(block=True)

    return cdf

def plot_population(env):
    """Plots population change over time steps.
    """
    seq = env.agent_count_seq
    print('Starting population: %i \nEnding population: %i' %(seq[0], seq[-1]))

    if isinstance(env, Sugarscape_Grow):
        metabolism = env.average_metabolism_seq
        print('Starting metabolism: %.2f \nEnding metabolism: %.2f' %(metabolism[0],
                                                                metabolism[-1]))

        vision = env.average_vision_seq
        print('Starting vision: %.2f \nEnding vision: %.2f' %(vision[0],
                                                                vision[-1]))

    plt.figure(figsize=(10, 7))

    if isinstance(env, Sugarscape_Grow):
        plt.subplot(3, 1, 1)

    plt.plot(seq, label='Population')
    decorate(xlabel='Time Steps', ylabel='Number of Agents')

    if isinstance(env, Sugarscape_Grow):
        plt.subplot(3,1,2)
        plt.plot(vision, label='Vision')
        decorate(ylabel='Vision')

        plt.subplot(3,1,3)
        plt.plot(metabolism, label='Metabolism')
        decorate(ylabel='Metabolism')

    plt.show(block=True)

def cdf_evolution(cdfs):
    """Plot CDF evolution over time.

    cdfs: list of Cdf
    """
    def plot_cdfs(cdfs, **options):
        for cdf in cdfs:
            cdf.plot(**options)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)

    plot_cdfs(cdfs[:-1], color='gray', alpha=0.3)
    plot_cdfs(cdfs[-1:], color='C0')
    decorate(xlabel='Wealth', ylabel='CDF')

    plt.subplot(1,2,2)
    plot_cdfs(cdfs[:-1], color='gray', alpha=0.3)
    plot_cdfs(cdfs[-1:], color='C0')
    decorate(xlabel='Wealth', ylabel='CDF', xscale='log')

    savefig('myfigs/chap09-4')
    plt.show(block=True)

if __name__ == '__main__':
    """Scripts go here"""
    ## Scripts
    np.random.seed(17)
    env = Sugarscape_Grow(50, num_agents=300)

    ani = env.animate()
    plt.show(block=True)

    plot_population(env)
