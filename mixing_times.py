from randomgraphs import GammaTBRW
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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

mixing_time_visualization()