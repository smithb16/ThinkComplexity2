
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

from itertools import islice
from utils import decorate, savefig


# TODO: remove this when NetworkX is fixed
from warnings import simplefilter
import matplotlib.cbook
simplefilter("ignore", matplotlib.cbook.mplDeprecation)

# node colors for drawing networks
colors = sns.color_palette('pastel', 5)
sns.set_palette(colors)

def undirected_graph():
	"""makes an undirected graph showing five cities and the drive times
	between them."""

	positions = dict(	Albany=(-74, 43),
						Boston=(-71, 42),
						NYC = (-74, 41),
						Philly = (-75, 40))

	G = nx.Graph()
	G.add_nodes_from(positions)

	drive_times = {('Albany','Boston'): 3,
					('Albany','NYC'): 4,
					('Boston', 'NYC'): 4,
					('NYC', 'Philly'): 2}

	G.add_edges_from(drive_times)

	positions['Pittsburgh'] = (-80, 40)

	new_drive_times = {('Pittsburgh', 'Albany'): 7,
						('Pittsburgh', 'NYC'): 6,
						('Pittsburgh', 'Philly'): 5,
						('Pittsburgh', 'Boston'): 9}

	G.add_node('Pittsburgh')
	G.add_edges_from(new_drive_times)

	drive_times = {**drive_times, **new_drive_times}

	nx.draw(G, positions,
			node_color='C1',
			node_shape='s',
			node_size=2500,
			with_labels=True)

	nx.draw_networkx_edge_labels(G, positions, edge_labels=drive_times)

	plt.show()

def directed_graph():
	"""makes a directed graph of a small directed social network (e.g. twitter)"""
	G = nx.DiGraph()

	G.add_node('Alice')
	G.add_node('Bob')
	G.add_node('Chuck')
	G.add_node('Ben')

	# syntax for directed graph: (<node one> points to <note two>)
	G.add_edge('Alice', 'Bob')
	G.add_edge('Alice', 'Chuck')
	G.add_edge('Bob', 'Alice')
	G.add_edge('Bob', 'Chuck')
	G.add_edge('Ben', 'Chuck')
	G.add_edge('Ben', 'Bob')
	G.add_edge('Alice', 'Ben')


	print(list(G.nodes()))

	print(list(G.edges()))

	nx.draw_circular(G,
					node_size = 2000,
					node_color = 'C0',
					with_labels = True)

	plt.axis('equal')
	plt.show()

def all_pairs(nodes):
	"""generator producing all pairs of tuples in range nodes
	nodes: iterable (e.g. range)"""

	for i, u in enumerate(nodes):
		for j, v in enumerate(nodes):
			if i>j:
				yield u, v

def m_pairs(nodes, m):
	"""returns a random sample of size m from the set all_pairs"""
	s = set(all_pairs(nodes))
	return random.sample(s, m)

def make_complete_graph(n):
	"""returns a complete undirected graph with n nodes"""
	G = nx.Graph()
	nodes = range(n)
	G.add_nodes_from(nodes)
	G.add_edges_from(all_pairs(nodes))
	return G

def reachable_nodes(G, start):
	"""returns the reachable nodes of G from start node.
	This algorithm performs depth-first-search and is inefficient
	for most uses.
	
	G: networkx.Graph or networkx.DiGraph
	start: node in G
	returns: list of nodes seen from start
	"""
	seen = set()
	stack = [start]
	while stack:
		node = stack.pop()
		if node not in seen:
			seen.add(node)
			stack.extend(G.neighbors(node))
	return seen

def is_connected(G):
	"""returns True if undirected graph G is fully connected, False otherwise
	"""
	start=next(iter(G))
	reachable = reachable_nodes_precheck(G, start)
	return len(reachable) == len(G)

def is_connected_directed(G):
	"""returns True if directed graph G is fully connected, False otherwise
	"""
	for node in G.nodes:
		# start = next(iter(G))
		start = node
		reachable = reachable_nodes(G, start)
		if len(reachable) != len(G):
			return False
	return True

def random_pairs(nodes, p):
	"""generates random edges based on flip of many-sided coin
	nodes: iterable of nodes (e.g. range)
	p: probability of returning node
	yields: edge
	"""
	for edge in all_pairs(nodes):
		if flip(p):
			yield edge

def flip(p):
	"""returns True if random float is less than p, false otherwise
	imitates flip of many-sided coin with p chance of success"""
	return np.random.random() < p

def make_random_graph(n, p):
	"""make random undirected graph
	n: number of nodes
	p: probability that a given edge exists
	returns: nx.Graph
	"""
	G = nx.Graph()
	nodes = range(n)
	G.add_nodes_from(nodes)
	G.add_edges_from(random_pairs(nodes, p))
	return G

def make_m_graph(n, m):
	"""makes random undirected graph of n nodes and m edges
	n: number of nodes
	m: number of edges
	returns: nx.Graph
	"""
	G = nx.Graph()
	nodes = range(n)
	G.add_nodes_from(nodes)
	G.add_edges_from(m_pairs(nodes, m))
	return G

def invert_tuple(g):
	"""generator that flips 2-value tuple output of another generator g
	"""
	while True:
		try:
			a, b = next(g)
			yield b, a
		except:
			break

def make_random_directed_graph(n,p):
	"""makes a random directed graph
	n: number of nodes in graph
	p: probability of a given node existing
	returns: nx.DiGraph
	"""
	G = nx.DiGraph()
	nodes = range(n)
	G.add_nodes_from(nodes)
	G.add_edges_from(random_pairs(nodes, p))

	reverse_pairs = invert_tuple( random_pairs(nodes, p) )
	G.add_edges_from(reverse_pairs)
	return G

def prob_connected(n, p=None, m=None, iters=100):
	"""determines the probabiltiy of an undirected Erdos-Renyi graph being connected.
	ER graph can be specified as ER(n,p) or ER(n,m), but not ER(n,p,m)
	n: number of nodes in graph
	p: probability of a given node existing
	m: number of edges in graph
	iters: number of iterations
	"""
	if (p==None) & (m==None):
		raise ValueError('prob_connected must have either p or m, but not both')
	
	elif (p!=None) & (m!=None):
		raise ValueError('prob_connected must have either p or m, but not both')

	elif p != None:
		tf = [is_connected(make_random_graph(n, p)) for i in range(iters)]

	elif m != None:
		tf = [is_connected(make_m_graph(n, m)) for i in range(iters)]

	return np.mean(tf)


if __name__ == '__main__':

	# undirected_graph()

	np.random.seed(17)

	nodes = range(10)
	print(m_pairs(nodes, 6))


	##############
	## Displays p(connected) vs. p(node) for various graph sizes
	# ns = [300, 100, 30, 10]
	# ps = np.logspace(-2.5, 0, 11)
	# sns.set_palette('Blues_r', 4)

	# for n in ns:

	# 	ys = [prob_connected(n, p=p) for p in ps]
	# 	pstar = np.log(n) / n
	# 	print(n)
	# 	plt.axvline(pstar, color='gray')
	# 	plt.plot(ps, ys, label='n=%d' %n)

	# decorate(xlabel='Prob of edge (p)',
	# 			ylabel = 'Prob connected',
	# 			xscale = 'log')

	##############
	## Displays p(connected) vs. m edges for various graph sizes
	# ns = [300, 100, 30, 10]
	# sns.set_palette('Blues_r', 4)

	# for n in ns:
	# 	print('n: ',n)
	# 	max_m = n*(n-1)/2
	# 	ps = np.logspace(-2.5, 0, 11)

	# 	ys = [prob_connected(n, m = int(p*max_m)) for p in ps]
	# 	pstar = np.log(n) / n
	# 	plt.axvline(pstar, color='gray')
	# 	plt.plot(ps, ys, label='n=%d' %n)

	# decorate(xlabel='Prob of edge (p)',
	# 			ylabel = 'Prob connected',
	# 			xscale = 'log')


	# random_graph = make_random_graph(10, 0.3)
	# print('is random directed graph connected?', end=' ')
	# print(is_connected( random_graph ))

	# nx.draw_circular(	G,
	# 					node_color='C2',
	# 					node_size=1000,
	# 					with_labels=True)
	
	# savefig('myfigs/chap02-6')
	# plt.show()