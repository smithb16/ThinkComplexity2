## Module import and boilerplate
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from empiricaldist import Cdf
from utils import decorate, savefig

## Function and class definition
class FitnessLandscape:
    """Maps genotype from each location in N-D space to a random fitness.
    """

    def __init__(self, N):
        """Creates a fitness landscape.

        N: int, number of dimensions
        """
        self.N = N
        self.set_values()

    def set_values(self):
        self.one_values = np.random.random(self.N)
        self.zero_values = np.random.random(self.N)

    def random_loc(self):
        """Choose a random location."""
        return np.random.randint(2, size=self.N, dtype=np.int8)

    def fitness(self, loc):
        """Evaluates the fitness of a location.

        loc: array of N 0s and 1s

        return: float fitness
        """
        fs = np.where(loc, self.one_values, self.zero_values)
        return fs.mean()

    def distance(self, loc1, loc2):
        return np.sum(np.logical_xor(loc1, loc2))

class Agent:
    """Represents an agent in an NK model."""

    def __init__(self, loc, fit_land):
        """Create an agent at the given location.

        loc: array of N 0s and 1s
        fit_land: reference to a FitnessLandscape
        """
        self.loc = loc
        self.fit_land = fit_land
        self.fitness = fit_land.fitness(self.loc)

    def copy(self):
        return Agent(self.loc, self.fit_land)

class Mutant(Agent):
    """Represents an agent with potential mutation."""

    def copy(self, prob_mutate=0.05):
        """Make a copy of a mutant.
        """
        if np.random.random() > prob_mutate:
            loc = self.loc.copy()
        else:
            direction = np.random.randint(self.fit_land.N)
            loc = self.mutate(direction)
        return Mutant(loc, self.fit_land)

    def mutate(self, direction):
        """Computes the location in the given direction.

        Result differs from current location along the given axis

        direction: int index from 0 to N-1

        return: new array of N 0s and 1s
        """
        new_loc = self.loc.copy()
        new_loc[direction] ^= 1
        return new_loc

class Simulation:

    def __init__(self, fit_land, agents):
        """Create the simulation:

        fit_land: FitnessLandscape
        agents: int number of agents
        agent_maker: function that makes agents
        """
        self.fit_land = fit_land
        self.agents = np.asarray(agents)
        self.instruments = []

    def add_instrument(self, instrument):
        """Adds an instrument to the list.

        instrument: Instrument object
        """
        self.instruments.append(instrument)

    def plot(self, index, *args, **kwargs):
        """Plot the results from the indicated instrument.
        """
        self.instruments[index].plot(*args, **kwargs)

    def run(self, num_steps=500):
        """Run the given number of steps.

        num_steps: int
        """
        ## initialize any instruments before starting
        self.update_instruments()

        for _ in range(num_steps):
            self.step()

    def step(self):
        """Simulate a time step and update the instruments.
        """
        n = len(self.agents)
        fits = self.get_fitnesses()

        # see who dies
        index_dead = self.choose_dead(fits)
        num_dead = len(index_dead)

        # replace the dead with copies of the living
        replacements = self.choose_replacements(num_dead, fits)
        self.agents[index_dead] = replacements

        # update any instruments
        self.update_instruments()

    def update_instruments(self):
        for instrument in self.instruments:
            instrument.update(self)

    def get_locs(self):
        """Return a list of agent locations."""
        return [tuple(agent.loc) for agent in self.agents]

    def get_fitnesses(self):
        """Return an array of agent fitnesses."""
        fits = [agent.fitness for agent in self.agents]
        return np.array(fits)

    def choose_dead(self, ps):
        """Choose which agents die in the next timestep.

        ps: probability of survival of each agent

        return: indices of the chosen ones
        """
        n = len(self.agents)
        is_dead = np.random.random(n) < 0.1
        index_dead = np.nonzero(is_dead)[0]
        return index_dead

    def choose_replacements(self, n, weights):
        """Choose which agents reproduce in the next timestep.

        n: int, number of choices
        weights: array of weights

        returns: sequence of Agent objects
        """
        agents = np.random.choice(self.agents, size=n, replace=True)
        replacements = [agent.copy() for agent in agents]
        return replacements

class Instrument:
    """Computes a metric at each timestep."""

    def __init__(self):
        self.metrics = []

    def update(self, sim):
        """Compute the current metric.

        Appends to self.metrics.

        sim: Simulation object
        """
        # child classes should implement this method
        pass

    def plot(self, **options):
        plt.plot(self.metrics, **options)

class MeanDistance(Instrument):
    """Computes the mean distance between pairs at each timestep."""
    label = 'Mean distance'

    def update(self, sim):
        N = sim.fit_land.N
        i, j = np.triu_indices(N)
        agents = zip(sim.agents[i], sim.agents[j])

        distances = [fit_land.distance(a1.loc, a2.loc) for a1, a2 in agents if a1 != a2]

        mean = np.mean(distances)
        self.metrics.append(mean)

class MeanFitness(Instrument):
    """Computes mean fitness at each timestep."""
    label = 'Mean fitness'

    def update(self, sim):
        mean = np.nanmean(sim.get_fitnesses())
        self.metrics.append(mean)

class OccupiedLocations(Instrument):
    """Computes the number of unique Agents."""
    label = 'Occupied locations'

    def update(self, sim):
        uniq_agents = len(set(sim.get_locs()))
        self.metrics.append(uniq_agents)

class SimWithDiffSurvival(Simulation):

    def choose_dead(self, ps):
        """Choose which agents die in the next timestep.

        ps: probability of survival for each agent

        return: indices of the chosen dead
        """
        n = len(self.agents)
        is_dead = np.random.random(n) > ps
        index_dead = np.nonzero(is_dead)[0]
        return index_dead

class SimWithDiffReproduction(Simulation):

    def choose_replacements(self, n, weights):
        """Choose which agents reproduce in the next timestep.

        n: int number of choices
        weights: array of weights

        return: sequence of Agent objects
        """
        p = weights / np.sum(weights)
        agents = np.random.choice(self.agents, size=n, replace=True, p=p)
        replacements = [agent.copy() for agent in agents]
        return replacements

class SimWithBoth(SimWithDiffSurvival, SimWithDiffReproduction):
    """Simulation with both differential survival and
    differential reproduction."""
    pass

def make_identical_agents(fit_land, num_agents, agent_maker):
    """Make an array of identical Agents.

    fit_land: FitnessLandscape
    num_agents: int
    agent_maker: class used to make Agent

    return: array of Agents
    """
    loc = fit_land.random_loc()
    agents = [agent_maker(loc, fit_land) for _ in range(num_agents)]
    return np.array(agents)

def make_random_agents(fit_land, num_agents, agent_maker):
    """Make an array of random Agents.

    fit_land: FitnessLandscape
    num_agents: int
    agent_maker: class used to make Agent

    return: array of Agents
    """
    locs = [fit_land.random_loc() for _ in range(num_agents)]
    agents = [agent_maker(loc, fit_land) for loc in locs]
    return np.array(agents)

def make_all_agents(fit_land, agent_maker):
    """Make an array of Agents.

    fit_land: FitnessLandscape
    agent_maker: class used to make Agent

    returns: array of Agents
    """
    N = fit_land.N
    locs = itertools.product([0, 1],repeat=N)
    agents = [agent_maker(loc, fit_land) for loc in locs]
    return np.array(agents)

def plot_fitnesses(sim):
    """Plot the CDF of fitnesses.

    sim: Simulation object
    """
    fits = sim.get_fitnesses()
    cdf_fitness = Cdf.from_seq(fits)
    print('Mean fitness\n', np.mean(fits))
    cdf_fitness.plot()
    decorate(xlabel='Fitness', ylabel='CDF')
    plt.show(block=True)

def plot_sims(fit_land, agent_maker, sim_maker, instrument_maker, **plot_options):
    """Runs simulations and plots metrics.

    fit_land: FitnessLandscape
    agent_maker: function that makes an array of Agent
    sim_maker: function that makes a Simulation
    instrument_maker: function that makes an Instrument
    plot_options: dict passed to plot
    """
    plot_options['alpha'] = 0.4

    for _ in range(10):
        agents = agent_maker(fit_land)
        sim = sim_maker(fit_land, agents)
        instrument = instrument_maker()
        sim.add_instrument(instrument)
        sim.run()
        sim.plot(index=0, **plot_options)

    decorate(xlabel='Time', ylabel=instrument.label)
    return sim

def agent_maker1(fit_land):
    return make_all_agents(fit_land, Agent)

def agent_maker2(fit_land):
    agents = make_identical_agents(fit_land, 100, Mutant)
    return agents

if __name__ == '__main__':
    """Script execution."""
    ## Scripts below here.
    np.random.seed(17)
    N = 10
    fit_land = FitnessLandscape(N)
    agents = agent_maker2(fit_land)
    sim = SimWithBoth(fit_land, agents)

    sim.add_instrument(MeanFitness())
    sim.add_instrument(OccupiedLocations())
    sim.add_instrument(MeanDistance())
    sim.run(500)
    locs_before = sim.get_locs()
    fit_land.set_values()
    sim.run(500)
    locs_after = sim.get_locs()

    vline_options = dict(color='gray', linewidth=2, alpha=0.4)

    sim.plot(2, color='C0')
    plt.axvline(500, **vline_options)
    decorate(xlabel='Time', ylabel='MeanFitness')
    plt.show(block=True)
