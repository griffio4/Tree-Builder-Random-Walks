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

def fast_tbrw_convergence_plot(iterations = 50, max_steps = 500, gammas=[1/2, 3/4, 1], include_loglog = True):
    for i, gamma in enumerate(gammas):
        distances_sum = np.zeros(max_steps)
        for j in range(iterations):
            distances = np.zeros(max_steps)
            tbrw = GammaTBRW(gamma)
            for k in range(max_steps):
                print(f"Gamma={round(gamma, 2)}, iteration {j}, step {k}          ", end="\r")
                deg = degrees(tbrw.T)
                degree_counts = np.zeros(max_steps)
                for d in deg:
                    degree_counts[d-1] += 1
                distances[k] = total_variational_distance(degree_counts / np.sum(degree_counts), power_law_ba(max_steps))
                tbrw.fast_growth() # technically fast_growth is not faster for gamma<.75, but we do this for all gammas for consistency
            distances_sum += distances
            if j == 0:
                plt.subplot(1 + int(include_loglog), len(gammas), i+1)
                plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
                if include_loglog:
                    plt.subplot(2, len(gammas), i+1+len(gammas))
                    plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
            else:
                plt.subplot(1 + int(include_loglog), len(gammas), i+1)
                plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))
                if include_loglog:
                    plt.subplot(2, len(gammas), i+1+len(gammas))
                    plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))

        plt.subplot(1 + int(include_loglog), len(gammas), i+1)
        plt.plot(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
        plt.xlabel("Tree size")
        plt.ylabel("Distance from power-law")
        plt.title(f"$\\gamma={round(gamma, 2)}$, linear")
        plt.legend()
        plt.grid()

        if include_loglog:
            plt.subplot(1 + int(include_loglog), len(gammas), i+1+len(gammas))
            plt.loglog(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
            plt.xlabel("Tree size")
            plt.ylabel("Distance from power-law")
            plt.title(f"$\\gamma={round(gamma, 2)}$, loglog")
            plt.legend()
            plt.grid()

    plt.subplots_adjust(wspace=.3)
    plt.show()

def tbrw_compare_methods(iterations = 100, max_steps = 500, gamma = 1/2):
    for i, update_func in enumerate(["Normal", "Fast"]):
        distances_sum = np.zeros(max_steps)
        for j in range(iterations):
            distances = np.zeros(max_steps)
            tbrw = GammaTBRW(gamma)
            for k in range(max_steps):
                print(f"update_func={update_func}, iteration {j}, step {k}          ", end="\r")
                deg = degrees(tbrw.T)
                degree_counts = np.zeros(max_steps)
                for d in deg:
                    degree_counts[d-1] += 1
                distances[k] = total_variational_distance(degree_counts / np.sum(degree_counts), power_law_ba(max_steps))
                if update_func == "Normal":
                    tbrw.update_to_size(tbrw.T.number_of_nodes() + 1)
                else:
                    tbrw.fast_growth()
            distances_sum += distances
            if j == 0:
                plt.subplot(2, 2, i+1)
                plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
                plt.subplot(2, 2, i+3)
                plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
            else:
                plt.subplot(2, 2, i+1)
                plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))
                plt.subplot(2, 2, i+3)
                plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))

        plt.subplot(2, 2, i+1)
        plt.plot(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
        plt.xlabel("Tree size")
        plt.ylabel("Distance from power-law")
        plt.title(f"{update_func} TBRW, linear")
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, i+3)
        plt.loglog(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
        plt.xlabel("Tree size")
        plt.ylabel("Distance from power-law")
        plt.title(f"{update_func} TBRW, loglog")
        plt.legend()
        plt.grid()

    plt.subplots_adjust(wspace=.3)
    plt.show()



def node_colors(count: int):
    color1 = (.4, .8, .8)
    color2 = (.8, .0, .0)
    tail = 10

    def mix(color1, color2, t):
        return (t * color1[0] + (1-t) * color2[0], t * color1[1] + (1-t) * color2[1], t * color1[2] + (1-t) * color2[2])
    
    c1 = {str(i) : color1 for i in range(max(0, count - tail))}
    c2 = {str(i + max(0, count - tail)) : mix(color1, color2, i / (tail-1)) for i in range(min(count, tail))}
    colors = dict(tuple(c1.items()) + tuple(c2.items()))
    print(colors)
    return colors