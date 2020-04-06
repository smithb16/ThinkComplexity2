import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

from scipy import spatial
from collections import deque
from utils import savefig, decorate

np.random.seed(17)

## TODO: remove this when networkx is fixed
from warnings import simplefilter
import matplotlib.cbook
simplefilter("ignore", matplotlib.cbook.mplDeprecation)

## node colors for drawing networks
colors = sns.color_palette('pastel', 5)
sns.set_palette(colors)

def adjacent_edges(nodes, halfk):
	"""Generates edges adjacent to node of form (source, dest)

	nodes: iterable of nodes (e.g. range)
	halfk: half of number of neighbors
	yields: edge
	"""
	n = len(nodes)
	for i, u in enumerate(nodes):
		for j in range(i+1, i+halfk+1):
			v = nodes[j % n]
			yield u,v

def opposite_edges(nodes):
	"""Generates edge crossing graph of form (source, dest)

	nodes: iterable of nodes (e.g. range)
	yields: edge
	"""
	n = len(nodes)
	print('n: ', n)
	halfnodes = nodes[0 : n//2]
	for i, u in enumerate(halfnodes):
		cross = (i+n//2) % n
		print(' u: ', u, ' cross: ', cross)
		yield u, nodes[cross]

def make_ring_lattice(n, k):
	"""Makes a ring lattice with `n` nodes and degree `k`.
	
	Note: this only works correctly if k is even.
	
	n: number of nodes
	k: degree of each node
	"""
	G = nx.Graph()
	nodes = range(n)
	G.add_nodes_from(nodes)
	G.add_edges_from(adjacent_edges(nodes, k//2))
	return G

def make_ws_graph(n, k, p):
	"""Makes a Watts-Strogatz graph
	n: int number of nodes
	k: int number of neighbors per node (degree of each node)
	p: probability of rewiring an edge
	"""
	ws = make_ring_lattice(n,k)
	rewire(ws, p)
	return ws

def rewire(G, p):
	"""Rewires each edge with probability 'p'

	G: nx.Graph
	p: float
	"""
	nodes = set(G)
	for u, v in G.edges():
		if flip(p):
			choices = nodes - {u} - set(G[u])
			new_v = np.random.choice(list(choices))
			G.remove_edge(u, v)
			G.add_edge(u, new_v)

def flip(p):
	"""returns True if random float is less than p, false otherwise
	imitates flip of many-sided coin with p chance of success"""
	return np.random.random() < p

def reachable_nodes_bfs(G, start):
	"""Performs breadth-first search of all nodes in 'G' 
	reachable from 'start' node.

	G: NetworkX Graph
	start: node

	returns: set of nodes seen from start node
	"""
	seen = set()
	queue = deque([start])
	while queue:
		node = queue.popleft()
		if node not in seen:
			seen.add(node)
			queue.extend(G.neighbors(node))
	return seen

def shortest_path_dijkstra(G, source):
	"""Finds shortest path from 'source' node to all other reachable
	nodes in 'G'.

	G: NetworkX.Graph
	source: node

	returns: dict of form {destination node: shortest distance}
	"""
	dist = {}
	new_dist = 0
	nextlevel = {source}

	while nextlevel:
		thislevel = nextlevel
		nextlevel = set()

		for v in thislevel:
			if v not in dist.keys():
				dist[v] = new_dist
				nextlevel.update(G[v])
		new_dist += 1
	return dist

def all_pairs_shortest_path(G):
	"""Implements Floyd-Warshal algorithm for unweighted,
	undirected graphs 'G'. Note that for unweighted undirected
	graphs, this is not more efficient than running Dijkstra n
	times.

	G: NetworkX.Graph
	returns: dist - np.matrix of distances between all vertices
	"""
	n = len(G.nodes())

	dist = np.full((n, n), np.inf)

	for u,v in G.edges():
		dist[u][v] = 1

	for v in G.nodes():
		dist[v][v] = 0

	for k in range(n):
		for i in range(n):
			for j in range(i,n):
				dist[j][i] = dist[i][j]

				if dist[i][j] > dist[i][k] + dist[k][j]:
					dist[i][j] = dist[i][k] + dist[k][j]

	print(dist)

	return dist


def dijkstra_dfs_bad(G, source):
	"""Attempts to perform dijkstra algorithm using depth-first search.
	This does not work.

	G: NetworkX.Graph
	source: node

	returns: dict of form {destination node: shortest distance}
	"""
	dist = {}
	new_dist = 0

	stack = [source]
	while stack:
		node = stack.pop()
		if node not in dist.keys():
			dist[node] = new_dist
			stack.extend(G.neighbors(node))

		new_dist += 1

	return dist

def node_clustering(G, u):
	"""Calculates the ratio of actual neighbors of node 'u'
	in networkx graph 'g' to possible neighbors.

	G: NetworkX graph
	u: node within G

	returns: ratio of neighbors to possible neighbors
	"""
	neighbors = G[u]
	k = len(neighbors)
	if k < 2:
		return np.nan

	possible = k * (k-1) / 2
	exist = 0
	for v, w in all_pairs(neighbors):
		if G.has_edge(v, w):
			exist += 1

	return exist/possible

def clustering_coefficient(G):
	"""Calculates average clustering coefficient of graph
	G: NetworkX Graph
	returns: float - mean clustering coefficient of each node
	"""
	cu = [node_clustering(G, node) for node in G]
	return np.nanmean(cu)

def plot_ws_exercise():
	n = 10
	k = 4
	ns = 100

	plt.subplot(1,4,1)
	ws = make_ws_graph(n, k, 0)
	nx.draw_circular(ws, node_size=ns)
	plt.axis('equal')

	plt.subplot(1,4,2)
	ws = make_ws_graph(n, k, 0.2)
	nx.draw_circular(ws, node_size=ns)
	plt.axis('equal')

	plt.subplot(1,4,3)
	ws = make_ws_graph(n, k, 0.4)
	nx.draw_circular(ws, node_size=ns)
	plt.axis('equal')

	plt.subplot(1,4,4)
	ws = make_ws_graph(n, k, 1.0)
	nx.draw_circular(ws, node_size=ns)
	plt.axis('equal')

	savefig('myfigs/chap03-2')
	plt.show()

def all_pairs(nodes):
	"""generator producing all pairs of tuples in range nodes
	nodes: iterable (e.g. range)"""

	for i, u in enumerate(nodes):
		for j, v in enumerate(nodes):
			if i>j:
				yield u, v

def path_lengths(G):
	"""Calculates shortest path length from all nodes to all other nodes in 'G'.
	G: NetworkX Graph

	returns: list of shortest path lengths
	"""
	length_map = dict(nx.shortest_path_length(G))
	lengths = [length_map[u][v] for u, v in all_pairs(G)]
	return lengths

def characteristic_path_length(G):
	""" Calculates average minimum path length between all node pairs.
	G: NetworkX Graph
	returns: float - mean shortest path length
	"""
	return np.mean(path_lengths(G))

def run_one_graph(n, k, p):
	"""Makes a WS graph and calculates its stats:

	n: number of nodes
	k: number of neighbors per node
	p: probability of node being rewired

	returns: tuple of (mean minimum path length, clustering coefficient)
	"""
	ws = make_ws_graph(n, k, p)
	mpl = characteristic_path_length(ws)
	cc = clustering_coefficient(ws)
	return mpl, cc

def run_one_graph_spatial(n, k, p):

	"""Makes a WS graph and calculates its stats:

	n: number of nodes
	k: number of neighbors per node
	p: probability of node being rewired

	returns: tuple of (mean minimum path length, clustering coefficient)
	"""
	sp, positions = make_spatial_graph(n, k, p)
	mpl = characteristic_path_length(sp)
	cc = clustering_coefficient(sp)
	return mpl, cc

def run_experiment(ps, n=1000, k=10, iters=20):
	"""Evaluates many WS graphs and calculates stats.

	ps: iterable (numpy array) of probabilities of rewiring node
	n: number of nodes
	k: number of neighbors per node
	iters: iterations at each probability

	returns: array of (mean min path length, clustering coefficient)
	"""
	res = []
	for p in ps:
		print(p)
		t=[run_one_graph(n, k, p) for _ in range(iters)]
		means = np.array(t).mean(axis=0)
		print(means)
		res.append(means)
	return np.array(res)

def run_experiment_spatial(ps, n=1000, k=10, iters=20):
	"""Evaluates many WS graphs and calculates stats.

	ps: iterable (numpy array) of probabilities of rewiring node
	n: number of nodes
	k: number of neighbors per node
	iters: iterations at each probability

	returns: array of (mean min path length, clustering coefficient)
	"""
	res = []
	for p in ps:
		print(p)
		t=[run_one_graph_spatial(n, k, p) for _ in range(iters)]
		means = np.array(t).mean(axis=0)
		print(means)
		res.append(means)
	return np.array(res)

def make_regular_graph(n, k):
	"""Makes a regular graph with `n` nodes and degree `k`.

	Calls make_ring_lattice if k is even.
	Raises ValueError if both n and k are odd.	

	n: number of nodes
	k: degree of each node
	"""

	if k%2 == 0:
		G = make_ring_lattice(n, k)

	elif (n%2 == 0) & (k%2 == 0):
		raise ValueError('Cannot make regular graph with odd values of n and k')

	else:
		G = make_ring_lattice(n, 2)
		nodes = range(n)
		G.add_edges_from(opposite_edges(nodes))

	return G

def make_spatial_graph(n, k, p):
	"""Makes a spatial graph 'n' nodes, each connected to its 'k' 
	closest neighbors with a random rewiring of edges in proportion to 'p'.
	In many cases, nodes will have >k neighbors but none will have less.

	n: int - number of nodes
	k: int - number of neighbors
	p: float - likelihood of rewiring an edge
	returns: 
		sp - graph of nodes
		positions - dictionary of node : (x,y)
		"""
	sp = nx.Graph()
	positions = {}
	for node in range(n):
		x = random.randint(0,1000)
		y = random.randint(0,1000)
		positions[node] = (x, y)

	sp.add_nodes_from(positions)
	tree = spatial.KDTree(list(positions.values()))

	for i in range(len(positions)):
		[dist, k_closest] = tree.query(tree.data[i], k+1)
		# print('i: ',i ,'\n',
		# 	'k_closest: ',k_closest)
		for j in range(1,k+1):
			sp.add_edge(k_closest[0], k_closest[j])

	rewire(sp, p)
	return sp, positions

def plot_ws_experiment(ps, data):
	L, C = np.transpose(data)
	print('L\n', L)
	print('C\n', C)
	L /= L[0]
	C /= C[0]

	plt.plot(ps, C, 's-', linewidth=1, label='C(p) / C(0)')
	plt.plot(ps, L, 'o-', linewidth=1, label='L(p) / L(0)')
	decorate(xlabel='Rewiring probability (p)', xscale='log',
			 title='Normalized clustering coefficient and path length',
			 xlim=[0.00009, 1.1], ylim=[-0.01, 1.01])

	savefig('myfigs/chap03-3')
	plt.show()

if __name__ == '__main__':


	G = make_ring_lattice(10, 4)
	all_pairs_shortest_path(G)
	# nx.draw(sp, positions,
	# 			 node_color='C0', 
	# 			 node_size=10, 
	# 			 with_labels=False)

	# plt.show()

	# Run Watt-Strogatz experiment
	# ps = np.logspace(-4, 0, 9)
	# data = run_experiment(ps)
	# print('data:\n',data)
	# plot_ws_experiment(ps, data)


	# nx.draw_circular(G, 
	# 			 node_color='C0', 
	# 			 node_size=1000, 
	# 			 with_labels=True)

	# savefig('myfigs/chap03-1')
	# plt.show()