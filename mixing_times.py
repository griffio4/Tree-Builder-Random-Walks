from randomgraphs import GammaTBRW
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def degrees(G: nx.Graph) -> np.array:
    return np.array([G.degree(v) for v in G.nodes])

def stationary_distribution(G: nx.Graph) -> np.array:
    return degrees(G) / np.sum(degrees(G))

def total_variational_distance(d1: np.array, d2: np.array):
    return np.sum(np.abs(d1 - d2)) / 2


def mixing_time_visualization():
    gamma = 1/2
    tree_size = 500
    walkers = 10000

    step_counts = [20, 100, 500, 2000]

    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    positions = tbrw.get_positions()
    tbrw.X = 0
    steps = 0
    walker_positions = np.zeros(walkers, dtype=np.int16)
    
    for i in range(len(step_counts)):
        plt.subplot(1,4,i+1)
        while steps < step_counts[i]:
            print(f"Simulating step {steps}         ", end="\r")
            for j in range(walkers):
                tbrw.X = walker_positions[j]
                tbrw.update_walk()
                walker_positions[j] = tbrw.X
            steps += 1
        walker_counts = np.zeros(tree_size, dtype=np.int16)
        for pos in walker_positions:
            walker_counts[pos] += 1
        transformed_data = np.arcsinh(10 * walker_counts) / np.arcsinh(10)
        nx.draw(tbrw.T, positions, node_color = transformed_data, node_size=30)
        plt.title(f"{step_counts[i]} steps")
        
    plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0, hspace=.1)
    plt.show()

def stationary_distance_visualization():
    gamma = 1/2
    tree_size = 500
    walkers_per_node = 500
    total_steps = 1000

    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    positions = tbrw.get_positions()
    tbrw.X = 0
    walker_positions = np.concatenate([i * np.ones(walkers_per_node, dtype=np.int16) for i in range(tree_size)])
    walkers = len(walker_positions)

    for i in range(total_steps):
        print(f"Simulating step {i}         ", end="\r")
        for j in range(walkers):
            tbrw.X = walker_positions[j]
            tbrw.update_walk()
            walker_positions[j] = tbrw.X
    
    walker_positions.reshape(walkers_per_node, tree_size)
    pi_T = stationary_distribution(tbrw.T)
    distances = np.array([total_variational_distance(walker_positions[i], pi_T) for i in range(tree_size)])

    nx.draw(tbrw.T, positions, node_color = distances, node_size=30)
    plt.show()


stationary_distance_visualization()

    
