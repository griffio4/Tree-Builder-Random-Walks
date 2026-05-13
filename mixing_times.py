from randomgraphs import GammaTBRW
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

gamma = 1/2
tree_size = 500
sample_size = 1000

step_counts = [3, 10, 30, 100, 300, 1000, 3000, 10000]

tbrw = GammaTBRW(gamma)
tbrw.update_to_size(tree_size, True)
positions = tbrw.get_positions()
tbrw.X = 0
steps = 0
walkers = np.zeros(sample_size)

times_visited = np.zeros(tree_size)
for i in range(len(step_counts)):
    plt.subplot(2,4,i+1)
    while steps < step_counts[i]:
        print(f"Simulating step {steps}         ", end="\r")
        for j in range(sample_size):
            tbrw.X = walkers[j]
            tbrw.update_walk()
            walkers[j] = tbrw.X
            times_visited[tbrw.X] += 1
        steps += 1
    transformed_data = np.arcsinh(10 * times_visited) / np.arcsinh(10)
    nx.draw(tbrw.T, positions, node_color = transformed_data, node_size=30)
    plt.title(f"{steps} steps")
    
plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0, hspace=.1)
plt.show()