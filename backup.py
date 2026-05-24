from randomgraphs import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# unused functions

def mixing_time_plot():
    gamma = 1/2
    tree_size = 100
    walkers = 10000
    step_counts = [100, 500, 2000]

    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    tbrw.X = 0
    steps = 0
    walker_positions = np.zeros(walkers, dtype=np.int16)
    sorted_vertices = list(sorted(tbrw.T.nodes, key=tbrw.T.degree))
    for i in range(len(step_counts)):
        while steps < step_counts[i]:
            print(f"Simulating step {steps}         ", end="\r")
            for j in range(walkers):
                tbrw.X = walker_positions[j]
                tbrw.update_walk()
                walker_positions[j] = tbrw.X
            steps += 1
        walker_counts = np.zeros(tree_size, dtype=np.int16)
        for pos in walker_positions:
            walker_counts[sorted_vertices.index(pos)] += 1
        i_ratio = i / (len(step_counts) - 1)
        plt.plot(np.arange(tree_size), walker_counts / walkers, color = (1 - i_ratio, .5, i_ratio), label=f"{steps} steps")
    degrees = np.array([tbrw.T.degree(v) for v in sorted_vertices])
    plt.plot(np.arange(tree_size), degrees / np.sum(degrees), color="orange", label="Stationary distribution")
    plt.legend()
    plt.xlabel("Vertices, sorted by degree")
    plt.ylabel("Proportion of walkers")
    plt.show()

def alpha_plot(gamma = 1/2, min_size = 50, max_size = 200, iterations=20):

    total_alphas = np.zeros(max_size - min_size)
    
    for i in range(iterations):
        alphas = np.zeros(max_size - min_size)
        tbrw = GammaTBRW(gamma)
        for j in range(min_size, max_size):
            print(f"Iteration {i}, size {j}        ", end="\r")
            tbrw.update_to_size(j)
            alphas[j - min_size] = mixing_constants(tbrw)[0]
        plt.plot(np.arange(min_size, max_size), alphas, color=(0.5,0.5,1.0,0.2))
        total_alphas += alphas
    plt.plot(np.arange(min_size, max_size), total_alphas / iterations, color="red")
    plt.show()