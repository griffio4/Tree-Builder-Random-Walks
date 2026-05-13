import networkx as nx
from random import choice
import numpy as np
from scipy.stats import rv_discrete
from typing import Callable
from random import random

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

    # graph drawing
    
    def draw(self):
        print("Drawing graph...       ")
        nx.draw(self.T, node_size=30)

    def draw_planar(self):
        print("Drawing graph...       ")
        nx.draw_planar(self.T, node_size=30)

    def draw_fancy(self):
        print("Drawing graph...       ")
        nx.draw_kamada_kawai(self.T, node_size=30)
    
    def export(self, path="tree.graphml"):
        print("Exporting graph...       ")
        nx.write_graphml(self.T, path)

    # analysis

    def degree_distribution(self):
        degrees = np.array([self.T.degree(v) for v in list(self.T.nodes)])
        return degrees / np.sum(degrees)
    

class TBRW(RandomTree):

    T: nx.Graph
    X: int
    n: int
    leaf_pmf: Callable
    
    def __init__(self, leaf_pmf: Callable, T0: nx.Graph = nx.path_graph(2), X0: int = 0):
        '''
        leaf_pmf: a function taking an integer n as input and returning a probability distribution (array)
        T0: the initial tree
        X0: the initial position of the walker
        '''
        if X0 not in T0.nodes:
            raise Exception(f"{X0} is not a vertex of the initial tree.")
        self.leaf_pmf = leaf_pmf
        self.T = T0.copy()
        self.X = X0
        self.n = 0
    
    def update_tree(self):
        # sample leaf count
        new_node_count = _sample_discrete(self.leaf_pmf(self.n))

        # add leaves to tree
        for _ in range(new_node_count):
            new_node = self.T.number_of_nodes()
            self.T.add_node(new_node)
            self.T.add_edge(self.X, new_node)
    
    def update_walk(self):
        self.X = choice(list(self.T.neighbors(self.X)))
    
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
    
    # fast update function
    def update_tree(self):
        if random() < self.n**(-self.gamma):
            new_node = self.T.number_of_nodes()
            self.T.add_node(new_node)
            self.T.add_edge(self.X, new_node)


class BA(RandomTree):
    T: nx.Graph
    n: int

    def update(self):
        print(f"Simulating n={self.n}", end="\r")

        # sample random vertex proportional to degree
        v = _sample_discrete(self.degree_distribution())

        # attach leaf
        new_node = new_node = self.T.number_of_nodes()
        self.T.add_node(new_node)
        self.T.add_edge(v, new_node)