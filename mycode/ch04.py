## Import modules
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

from empiricaldist import Pmf, Cdf
from networkx.algorithms.approximation import average_clustering
from collections import deque
from utils import savefig, decorate

np.random.seed(17)

# TODO: remove this when networkx is fixed
from warnings import simplefilter
import matplotlib.cbook
simplefilter("ignore", matplotlib.cbook.mplDeprecation)

# node colors for drawing networks
colors = sns.color_palette('pastel', 5)
sns.set_palette(colors)

## Function definitions
"""Reads data file and generates network.
filename: file with network data
n: maximum number of lines to read
    Used to limit network size in giant files.
returns: G - NetworkX graph
"""
def read_actor_network(filename, n=None):
    G = nx.Graph()
    with gzip.open(filename) as f:
        for i, line in enumerate(f):
            nodes = [int(x) for x in line.split()]
            G.add_edges_from(all_pairs(nodes))
            if n and (i>=n):
                break

    return G

def all_pairs(nodes):
    """Generates all pairs of nodes."""
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v

def read_graph(filename):
    """Reads filename and generates nx.Graph
    """
    G = nx.Graph()
    array = np.loadtxt(filename, dtype=int)
    G.add_edges_from(array)
    return G

def sample_path_lengths(G, nodes=None, trials=1000):
    """Measures characteristic shortest path length of 'G' via sampling
    of random pairs within 'nodes'. Uses all nodes of G if not specified.

    G: NetworkX.Graph
    nodes: set of nodes to sample for characteristic path length
    trials: int - number of trials to run

    returns: list of lengths
    """
    if nodes is None:
        nodes = list(G)
    else:
        nodes = list(nodes)

    pairs = np.random.choice(nodes, (trials, 2))
    lengths = [nx.shortest_path_length(G, *pair)
                for pair in pairs]

    return lengths

def estimate_path_length(G, nodes=None, trials=1000):
    """Measures characteristic path length of graph 'G'
    via random sampling.
    G: NetworkX graph
    nodes: set of nodes to sample for characteristic path length
    trials: int - number of trials to run
    returns: mean path length - float
    """
    return np.mean(sample_path_lengths(G, nodes, trials))

def degrees(G):
    """Returns list of number of degrees of each node
    in NetworkX Graph 'G'
    """
    return[G.degree(u) for u in G]

def analyze_graph(G, verbose=False):
    """Analyzes 'G' for relevant characteristics.
    G: NetworkX graph
    verbose: print characteristics if True
    return:
        n: number of nodes in G
        m: number of edges in G
        k: int of average degree (edges per node)
        degs: list of degrees of G
        """
    n = len(G)
    m = len(G.edges())
    k = int(round(m/n))
    C = average_clustering(G)
    L = estimate_path_length(G)
    degs = degrees(G)

    if verbose:
        print('n: %i  m: %i  k: %i' %(n,m,k))
        print('clustering: ',C, 'path length: ',L)
        print('average degree: %.2f  degree variance: %.2f'
                %(np.mean(degs), np.var(degs)))

    return n, m, k, degs

def compare_fb_to_ws():
    """Plots Facebook network data vs. Watts-Strogatz
    """
    dirname = '/Users/bensmith/Documents/ThinkSeries/ThinkComplexity2/data/'
    fin = dirname + 'facebook_combined.txt.gz'
    fb = read_graph(fin)

    print('Facebook')
    n, m, k, degs = analyze_graph(fb)
    pmf_fb = Pmf.from_seq(degs)

    x = 25
    print('fewer than %i friends: %.3f' %(x, cumulative_prob(pmf_fb, x)))

    ws = nx.watts_strogatz_graph(n, k, 0.05, seed=15)
    print('Watts-Strogatz')
    n, m, k, degs = analyze_graph(ws)
    pmf_ws = Pmf.from_seq(degs)

    plt.figure(figsize=(8,4))
    options = dict(ls='', marker='.')


    plt.subplot(1,2,1)
    plt.plot([20, 1000], [5e-2, 2e-4], color='gray', linestyle='dashed')
    pmf_fb.plot(label='Facebook', color='C0', **options)
    decorate(xlabel='Degree', ylabel='PMF',
                xscale='log', yscale='log')

    plt.subplot(1,2,2)
    pmf_ws.plot(label='WS graph', color='C1', **options)
    decorate(xlabel='Degree',
                xscale='log', yscale='log')

    savefig('myfigs/chap04-1')
    plt.show()

def compare_fb_to_ba():
    """Plots Facebook network data vs. Barabasi-Albert network of same size"""

    dirname = '/Users/bensmith/Documents/ThinkSeries/ThinkComplexity2/data/'
    fin = dirname + 'facebook_combined.txt.gz'
    fb = read_graph(fin)

    print('Facebook')
    n, k, pmf_fb = analyze_graph(fb)

    ba = barabasi_albert_graph(n, k, seed=15)
    print('Barabasi-Albert')
    n, k, pmf_ba = analyze_graph(ba)

    plt.figure(figsize=(8,4))
    options = dict(ls='', marker='.')


    plt.subplot(1,2,1)
    plt.plot([20, 1000], [5e-2, 2e-4], color='gray', linestyle='dashed')
    pmf_fb.plot(label='Facebook', color='C0', **options)
    decorate(xlabel='Degree', ylabel='PMF',
                xscale='log', yscale='log')

    plt.subplot(1,2,2)
    pmf_ba.plot(label='BA graph', color='C1', **options)
    decorate(xlabel='Degree',
                xscale='log', yscale='log')

    savefig('myfigs/chap04-2')
    plt.show()

def barabasi_albert_graph(n, k, seed=None):
    """Generate Barabasi-Albert graph with 'n' nodes with an
    average degree 'k'.
    n: number of nodes
    k: average degree of nodes
    seed: seed for random operation
    returns: G - NetworkX Graph
    """
    if seed is not None:
        seed = random.seed(seed)

    G = nx.empty_graph(k)
    targets = set(range(k))
    repeated_nodes = []

    for source in range(k, n):
        G.add_edges_from(zip([source]*k, targets))

        repeated_nodes.extend(targets)
        repeated_nodes.extend([source]*k)

        targets = _random_subset(repeated_nodes, k)

    return G

def _random_subset(repeated_nodes, k):
    """Returns a random set of size 'k' from
    options in 'repeated_nodes'.
    repeated_nodes: list of choices
    k: int of size of returned set
    targets: set of nodes
    """
    targets = set()
    while len(targets) < k:
        x = random.choice(repeated_nodes)
        targets.add(x)

    return targets

def generate_hk_graph(n, k, p, seed=None):
    """Generate network using Holme-Kim algorithm.
    n: number of nodes
    k: average degree of node
    p: probability of forming triangle during implementation of B-A algorithm
    seed: int used to seed random operations
    return: G - NetworkX Graph
    """
    if seed is not None:
        seed=random.seed(seed)

    G = nx.powerlaw_cluster_graph(n, k, p, seed)

    return G

def combine_analyze_graph(s):
    """Performs function of analyze_graph on list of graphs 's'.
    s: list of NetworkX Graphs
    returns:
        n: combined number of nodes
        m: combined number of edges
        k: average edges per node
        pmf_full: pmf of degrees of edges
        """
    n = 0
    m = 0
    d = []
    for g in s:
        n_s, m_s, k_s, degs = analyze_graph(g)
        n += n_s
        m += m_s
        d += degs

    k = n/m
    return n, m, k, sorted(d)

if __name__ == '__main__':
    """Runs various functions. Split into cells for cell-wise execution
    if applicable.
    """
    ## Read data from Facebook file
    dirname = '/Users/bensmith/Documents/ThinkSeries/ThinkComplexity2/data/'
    fin = dirname + 'facebook_combined.txt.gz'
    fb = read_graph(fin)

    n, m, k, pmf_fb = analyze_graph(fb, verbose=True)
    print('pmf_fb:\n',type(pmf_fb))

    ## Build ws & ba models that closely represent Facebook data
    ws = nx.watts_strogatz_graph(n, k, 0.05, seed=15)
    ba = nx.barabasi_albert_graph(n, k, seed=15)
    hk = generate_hk_graph(n, k, 1, seed=15)

    ## Generate CDFs of three graphs
    cdf_fb = Cdf.from_seq(degrees(fb), name='Facebook')
    cdf_ws = Cdf.from_seq(degrees(ws), name='Watts-Strogatz')
    cdf_ba = Cdf.from_seq(degrees(ba), name='Barabasi-Albert')
    cdf_hk = Cdf.from_seq(degrees(hk), name='Holme-Kim')

    ## Generate HK graph that mimics Facebook data
    ps = np.logspace(-4, 0, 9)

    for p in ps:
        G = generate_hk_graph(n, k, p)
        print('\np: ',p)
        n, m, k, pmf_hk = analyze_graph(G, verbose=True)

    ## Generate figures comparing degree of facebook to degree of WS & BA models
    plt.figure(figsize=(8,4))

    plt.subplot(1,3,1)
    (1-cdf_fb).plot(color='C0')
    (1-cdf_hk).plot(color='C1', alpha=0.4)
    decorate(xlabel='Degree', xscale='log',
                     ylabel='CCDF', yscale='log')

    plt.subplot(1,3,2)
    (1-cdf_fb).plot(color='C0')
    (1-cdf_ws).plot(color='C1', alpha=0.4)
    decorate(xlabel='Degree', xscale='log',
                     ylabel='CCDF', yscale='log')

    plt.subplot(1,3,3)
    (1-cdf_fb).plot(color='C0', label='Facebook')
    (1-cdf_ba).plot(color='C2', alpha=0.4)
    decorate(xlabel='Degree', xscale='log',
                    ylabel='CCDF', yscale='log')

    savefig('myfigs/chap04-7')
    plt.show()

    ## Read actor data
    dirname = '/Users/bensmith/Documents/ThinkSeries/ThinkComplexity2/data/'
    fin = dirname + 'actor.dat.gz'
    act_net = read_actor_network(fin)
    print('Data read')

    s_networks = [act_net.subgraph(c).copy() for c in nx.connected_components(act_net)]
    print('Number of subgraphs generated.', len(s_networks))

    ## Analyze combined network
    n, m, k, degs = combine_analyze_graph(s_networks)
    pmf_act = Pmf.from_seq(degs)
    print('Graph analyzed.')

    ## Make CDF
    cdf = Cdf.from_seq(degs, name='Actors')

    ## Plot CDF & CCDF
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    cdf.plot(color='C0', label='Actors')
    decorate(xlabel='Degree', xscale='log',
                    ylabel='CDF')

    plt.subplot(1,2,2)
    (1-cdf).plot(color='C0')
    decorate(xlabel='Degree', xscale='log',
                     ylabel='CCDF', yscale='log')
    savefig('myfigs/chap04-8')
    plt.show()
