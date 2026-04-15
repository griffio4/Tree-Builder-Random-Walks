import networkx as nx
from random import choice
import numpy as np
from scipy.stats import rv_discrete
from typing import Callable
import matplotlib.pyplot as plt

class TBRW:

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
        self.T = T0
        self.X = X0
        self.n = 0
    
    def update(self):

        print(f"Simulating n={self.n}", end="\r")

        # sample leaf count
        pmf = self.leaf_pmf(self.n)
        new_node_count = rv_discrete(values=(np.arange(len(pmf)), pmf)).rvs(size=1)[0]

        # add leaves to tree
        for _ in range(new_node_count):
            new_node = self.T.number_of_nodes()
            self.T.add_node(new_node)
            self.T.add_edge(self.X, new_node)

        # move walker
        self.X = choice(list(self.T.neighbors(self.X)))

        self.n += 1      
    
    def repeat_update(self, iterations: int):
        for i in range(iterations):
            self.update()
    
    def update_to(self, end_n: int):
        self.repeat_update(end_n - self.n)
        
    def update_to_size(self, end_size: int):
        while self.T.number_of_nodes() < end_size:
            self.update()
    
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


leaf_pmf = lambda n: np.array([.9, .1])
tbrw = TBRW(leaf_pmf)

tbrw.update_to(5000)
tbrw.draw_fancy()

plt.show()
