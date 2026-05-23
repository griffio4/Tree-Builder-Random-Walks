from randomgraphs import GammaTBRW
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm, colors


def degrees(G: nx.Graph) -> np.array:
    return np.array([G.degree(v) for v in G.nodes])

def stationary_distribution(G: nx.Graph) -> np.array:
    return degrees(G) / np.sum(degrees(G))

def total_variational_distance(d1: np.array, d2: np.array):
    return np.sum(np.abs(d1 - d2)) / 2


def mixing_visualization():
    gamma = 1/2
    tree_size = 500

    step_counts = [10, 100, 500, 2000]

    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    positions = tbrw.get_positions()
    transition_matrix = tbrw.transition_matrix()
    
    for i, steps in enumerate(step_counts):
        plt.subplot(1, len(step_counts), i+1)
        P = np.zeros(tree_size)
        P[0] = 1
        P @= np.linalg.matrix_power(transition_matrix, steps)
        P_transformed = np.log(P + 10**-6)
        nx.draw(tbrw.T, positions, node_color = P_transformed, node_size=30)
        plt.title(f"{steps} steps")
    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0, hspace=.1)
    plt.show()

def stationary_distance_visualization():
    gamma = 1/2
    tree_size = 1000
    steps = 2000

    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    pos1 = tbrw.get_positions()
    pos2 = tbrw.get_positions(layout="dot")
    pi_T = stationary_distribution(tbrw.T)

    distributions = np.linalg.matrix_power(tbrw.transition_matrix(), steps)
    distances = np.array([total_variational_distance(distributions[i], pi_T) for i in range(tree_size)])

    plt.subplot(121)
    nx.draw(tbrw.T, pos1, node_color = distances, node_size=30)
    plt.subplot(122)
    nx.draw(tbrw.T, pos2, node_color = distances, node_size=30)

    norm = colors.Normalize(min(distances), max(distances))
    plt.colorbar(cm.ScalarMappable(norm=norm), shrink=.8)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0, hspace=0)
    plt.show()

stationary_distance_visualization()

    
