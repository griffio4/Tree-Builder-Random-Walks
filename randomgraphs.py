import networkx as nx
from random import choice
import numpy as np
from scipy.stats import rv_discrete
from typing import Callable
from random import random
from networkx.drawing.nx_agraph import graphviz_layout

def _sample_discrete(pmf):
    return rv_discrete(values=(np.arange(len(pmf)), pmf)).rvs(size=1)[0]

class RandomTree:
    T: nx.graph
    n: int

    def __init__(self, T0: nx.Graph = nx.path_graph(2)):
        self.T = T0.copy()
        self.n = 1
    
    # updating functions

    def update(self):
        pass

    def update_log(self, log=False):
        if log:
            print(f"Simulating n={self.n}, vertex count {self.T.number_of_nodes()}", end="\r")
        self.update()

    def repeat_update(self, iterations: int, log=False):
        for i in range(iterations):
            self.update_log(log)
        print("")
    
    def update_to(self, end_n: int, log=False):
        self.repeat_update(end_n - self.n, log)
        
    def update_to_size(self, end_size: int, log=False):
        while self.T.number_of_nodes() < end_size:
            self.update_log(log)
        if log:
            print("")

    def export(self, path="tree.graphml"):
        print("Exporting graph...                  ")
        nx.write_graphml(self.T, path)
        print("Finished exporting graph.")
    
    def get_positions(self, layout="sfdp"):
        print("Computing vertex positions...       ")
        pos = graphviz_layout(self.T, prog=layout)
        print("Finished computing vertex positions.")
        return pos
        

    def degree_distribution(self):
        degrees = np.array([self.T.degree(v) for v in list(self.T.nodes)])
        return degrees / np.sum(degrees)


class TBRW(RandomTree):

    T: nx.Graph
    X: int
    n: int
    leaf_pmf: Callable
    selfloop: bool
    
    def __init__(self, leaf_pmf: Callable, T0: nx.Graph = nx.path_graph(2), X0: int = 0):
        # leaf_pmf is a function taking a parameter n (int) and returning a probability distribution (np.array)
        if X0 not in T0.nodes:
            raise Exception(f"{X0} is not a vertex of the initial tree.")
        self.leaf_pmf = leaf_pmf
        self.T = T0.copy()
        self.X = X0
        self.n = 0
        self.selfloop = True
    
    def update_tree(self):
        # sample leaf count
        new_node_count = _sample_discrete(self.leaf_pmf(self.n))

        # add leaves to tree
        for _ in range(new_node_count):
            new_node = self.T.number_of_nodes()
            self.T.add_node(new_node)
            self.T.add_edge(self.X, new_node)
    
    def update_walk(self):
        if self.X == 0 and random() < 1/(self.T.degree(0) + 1): # self-loop
            return
        self.X = choice(list(self.T.neighbors(self.X)))
    
    def transition_vector(self, x) -> np.array:
        vec = np.zeros(self.T.number_of_nodes())
        for n in self.T.neighbors(x):
            vec[n] = 1
        if x == 0 and self.selfloop: # self-loop
            vec[0] = 1
            return vec / (self.T.degree(x) + 1)
        return vec / self.T.degree(x)
        
    def transition_matrix(self) -> np.array:
        return np.array([self.transition_vector(x) for x in range(self.T.number_of_nodes())], dtype=np.float64)
    
    def update(self):
        self.update_tree()
        self.update_walk()
        self.n += 1      

    def growth_times(self, end, log=False) -> np.array:
        times = []
        for i in range(1, end + 1):
            self.update_to_size(i, log)
            times.append(self.n)
        return np.array(times)


class GammaTBRW(TBRW):

    gamma: int
    
    def __init__(self, gamma: int, T0: nx.Graph = nx.path_graph(2), X0: int = 0):
        self.T = T0.copy()
        self.X = X0
        self.n = 1
        self.gamma = gamma
    
    # faster tree update function
    def update_tree(self):
        if random() < self.n**(-self.gamma):
            new_node = self.T.number_of_nodes()
            self.T.add_node(new_node)
            self.T.add_edge(self.X, new_node)
    
    # fast approximate growth using matrix multiplication, see Section 2.7
    def fast_growth(self):
        P = self.transition_matrix()

        # inverse CDF as in Section 2.7
        def inverse_cdf(x, gamma, tau_n):
            if gamma <= 0 or gamma > 1:
                raise Exception(f"{gamma} is an invalid value of gamma.")
            if gamma == 1:
                return tau_n * (1/(1-x)-1)
            else:
                return ((gamma-1)*np.log(1-x) + tau_n**(1-gamma))**(1/(1-gamma)) - tau_n
        
        # determine distribution of X after growth time
        x = random()
        growth_time = inverse_cdf(x, self.gamma, self.n)
        if growth_time < 0:
            raise Exception(f"Growth time {growth_time} is negative. Values: x={x}, gamma={self.gamma}, tau_n={self.n}")
        growth_time = round(growth_time)
        dist = np.zeros(self.T.number_of_nodes())
        dist[self.X] = 1
        dist @= np.linalg.matrix_power(P, growth_time)

        # sample X from distribution
        self.X = _sample_discrete(dist)

        # attach leaf
        new_node = self.T.number_of_nodes()
        self.T.add_node(new_node)
        self.T.add_edge(self.X, new_node)


class BA(RandomTree):
    T: nx.Graph
    n: int

    def update(self):
        # sample random vertex proportional to degree
        v = _sample_discrete(self.degree_distribution())

        # attach leaf
        new_node = new_node = self.T.number_of_nodes()
        self.T.add_node(new_node)
        self.T.add_edge(v, new_node)

def degrees(G: nx.Graph) -> np.array:
    deg = np.array([G.degree(v) for v in G.nodes])
    deg[0] += 1 # self-loop
    return deg

def stationary_distribution(G: nx.Graph) -> np.array:
    return degrees(G) / np.sum(degrees(G))

def total_variational_distance(d1: np.array, d2: np.array):
    return np.sum(np.abs(d1 - d2)) / 2