## Module import and boilerplate
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from pandas import Series
from utils import decorate, savefig
from ch11 import Simulation, Instrument, MeanFitness, SimWithDiffReproduction

## Class and function definition
class Agent:
    """Defines an Agent in a Prisoner's Dilemma evolution
    """
    keys = [(None, None),
            (None, 'C'),
            (None, 'D'),
            ('C', 'C'),
            ('C', 'D'),
            ('D', 'C'),
            ('D', 'D')]

    def __init__(self, values, fitness=np.nan):
        """Initialize the agent.

        values: sequence of 'C' and 'D'
        """
        self.values = values
        self.responses = dict(zip(self.keys, values))
        self.fitness = fitness

    def reset(self):
        """Reset the variables before a sequence of games.
        """
        self.hist = [None, None]
        self.score = 0

    def past_responses(self, num=2):
        """Select the given number of most recent responses.

        num: integer number of responses

        return: sequence of 'C' and 'D'
        """
        return tuple(self.hist[-num:])

    def respond(self, other):
        """Choose a response based on the opponent's recent responses.

        other: Agent

        returns: 'C' or 'D'
        """
        key = other.past_responses()
        resp = self.responses[key]
        return resp

    def append(self, resp, pay):
        """Update based on the last response and payoff

        resp: 'C' or 'D'
        pay: int
        """
        self.hist.append(resp)
        self.score += pay

    def copy(self, prob_mutate=0.01):
        """Make a copy of this agent.
        """
        if np.random.random() > prob_mutate:
            values = self.values
        else:
            values = self.mutate()
        return Agent(values, self.fitness)

    def mutate(self):
        """Makes a copy of this agent's values, with one mutation.

        returns: sequence of 'C' and 'D'
        """
        values = list(self.values)
        index = np.random.choice(len(values))
        values[index] = 'C' if values[index] == 'D' else 'D'
        return values

class Tournament:

    payoffs = {('C','C'):(3,3),
                ('C','D'):(0,5),
                ('D','C'):(5,0),
                ('D','D'):(1,1)}

    num_rounds = 6

    def play(self, agent1, agent2):
        """Play a sequence of iterated PD rounds.

        agent1: Agent
        agent2: Agent

        return: tuple of agent1 score, agent2 score
        """
        agent1.reset()
        agent2.reset()

        for i in range(self.num_rounds):
            resp1 = agent1.respond(agent2)
            resp2 = agent2.respond(agent1)

            pay1, pay2 = self.payoffs[resp1, resp2]

            agent1.append(resp1, pay1)
            agent2.append(resp2, pay2)

        return agent1.score, agent2.score

    def melee(self, agents, randomize=True, num_opponents=2):
        """Play each agent against two others.

        Assigns the average score from two games to agent.fitness

        agents: sequence of Agents
        randomize: boolean, whether to shuffle the agents
        num_opponents: int, number of opponents each agent faces
                        must be even number
        """
        assert num_opponents % 2 == 0, 'num_opponents must be even'

        if randomize:
            agents = np.random.permutation(agents)

        [agent.reset() for agent in agents]

        n = len(agents)
        matches = []
        totals = {}

        # Build the list of matches
        for i, opp1 in enumerate(agents):
            totals[opp1] = 0
            for j in range(num_opponents//2):
                opp2 = agents[(i+j+1) % n]
                matches.append((opp1, opp2))

        for agent1, agent2 in matches:
            score1, score2 = self.play(agent1, agent2)

            totals[agent1] += score1
            totals[agent2] += score2

        for agent, score in totals.items():
            agent.fitness = score / self.num_rounds / num_opponents

class PDSimulation(Simulation):

    def __init__(self, tournament, agents):
        """Creates the simulation.

        tournament: Tournament object
        agents: sequence of agents
        """
        self.tournament = tournament
        self.agents = np.asarray(agents)
        self.instruments = []

    def step(self):
        """Simulate a time step and update the instruments.
        """
        self.tournament.melee(self.agents, randomize=True, num_opponents=2)
        Simulation.step(self)

    def choose_dead(self, fits):
        """Choose which agents die in the next timestep.

        fits: fitness of each agent

        return: indices of the chosen dead
        """
        ps = prob_survive(fits)
        n = len(self.agents)
        is_dead = np.random.random(n) < ps
        index_dead = np.nonzero(is_dead)[0]
        return index_dead

class PDSimWithDiffReproduction(PDSimulation):
    """Prisoner's dilemma simulation with differential reproduction.
    """
    def step(self):
        """Simulate a time step and update instruments.
        """
        self.tournament.melee(self.agents, randomize=True, num_opponents=2)
        SimWithDiffReproduction.step(self)

class Niceness(Instrument):
    """Fraction of cooperation in all genotypes."""
    label = 'Niceness'

    def update(self, sim):
        responses = np.array([agent.values for agent in sim.agents])
        metric = np.mean(responses == 'C')
        self.metrics.append(metric)

class Niceness2(Instrument):
    """Fraction of cooperation in last four elements of genotypes."""
    label = 'Niceness - Last four genes'

    def update(self, sim):
        responses = np.array([agent.values[-4:] for agent in sim.agents])
        metric = np.mean(responses == 'C')
        self.metrics.append(metric)

class Opening(Instrument):
    """Fraction of agents that cooperate on the first round."""
    label = 'Opening'

    def update(self, sim):
        responses = np.array([agent.values[0] for agent in sim.agents])
        metric = np.mean(responses == 'C')
        self.metrics.append(metric)

class Retaliating(Instrument):
    """Tendency to defect after opponent defects."""
    label = 'Retaliating'

    def update(self, sim):
        after_d = np.array([agent.values[2::2] for agent in sim.agents])
        after_c = np.array([agent.values[1::2] for agent in sim.agents])
        metric = np.mean(after_d == 'D') - np.mean(after_c == 'D')
        self.metrics.append(metric)

class Forgiving(Instrument):
    """Tendency to cooperate if the opponent cooperates after defecting."""
    label = 'Forgiving'

    def update(self, sim):
        after_dc = np.array([agent.values[5] for agent in sim.agents])
        after_cd = np.array([agent.values[4] for agent in sim.agents])
        metric = np.mean(after_dc == 'C') - np.mean(after_cd == 'C')
        self.metrics.append(metric)

class Forgiving2(Instrument):
    """Ability to cooperate after the first two rounds."""
    label = 'Forgiving2'

    def update(self, sim):
        after_two = np.array([agent.values[3:] for agent in sim.agents])
        metric = np.mean(np.any(after_two=='C', axis=1))
        self.metrics.append(metric)

def make_identical_agents(n, values):
    """Make agents with the given genotype

    n: number of agents
    values: sequence of 'C' and 'D'

    return: sequence of agents
    """
    agents = [Agent(values) for _ in range(n)]
    return agents

def make_random_agents(n):
    """Makes agents with randomm genotype.

    n: number of agents
    return: sequence of agents
    """
    agents = [Agent(np.random.choice(['C', 'D'], size=7)) for _ in range(n)]
    return agents

def logistic(x, A=0, B=1, C=1, M=0, K=1, Q=1, nu=1):
    """Computes the generalized logistic function.

    A: controls the lower bound
    B: controls the steepness of the transition
    C: not all that useful AFAIK
    M: controls the location of the transition
    K: controls the upper bound
    Q: shift the transition left or right
    nu: affects the symmetry of the transition

    return: float or array
    """
    exponent = -B * (x - M)
    denom = C + Q * np.exp(exponent)
    return A + (K-A) / denom ** (1/nu)

def prob_survive(scores):
    """Probability of survival, based on fitness.

    scores: sequence of scores, 0-5
    returns: probability
    """
    return logistic(scores, A=0.7, B=1.5, M=2.5, K=0.9)

def plot_result(index, **options):
    """Plot the results of the indicated instrument.

    index: int
    """
    sim.plot(index, **options)
    instrument = sim.instruments[index]
    print('Mean ', instrument.label, np.mean(instrument.metrics[:]))
    decorate(xlabel='Time steps', ylabel=instrument.label)


## Scripts below here
if __name__ == '__main__':

    responses = []
    final_fits = []

    ## See which genotypes are most resilient
    for i in range(20):
        tour = Tournament()
        agents = make_random_agents(100)
        sim = PDSimulation(tour, agents)
        sim.run(500)
        fit = np.mean(sim.get_fitnesses())
        final_fits.append(fit)

        responses += [''.join(agent.values) for agent in sim.agents]

    ## Show genomes
    print(Series(responses).value_counts())

    ## Plot final fitnesses
    plt.plot(final_fits, label='Final fitness')
    decorate(ylim=[0,5.1], xlabel='Simulation', ylabel='Final fitness')
    plt.show(block=True)
