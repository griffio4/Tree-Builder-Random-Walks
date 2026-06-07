from randomgraphs import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def power_law_ba(size: int) -> np.array:
    d = np.arange(size) + 1 # start indexing at 1 to prevent division by zero
    return 4 / (d * (d+1) * (d+2))

def ba_convergence_plot(iterations = 50, max_steps = 3000):
    distances_sum = np.zeros(max_steps)
    for i in range(iterations):
        distances = np.zeros(max_steps)
        ba = BA()
        for j in range(max_steps):
            print(f"Iteration {i}, step {j}          ", end="\r")
            deg = degrees(ba.T)
            degree_counts = np.zeros(max_steps)
            for d in deg:
                degree_counts[d-1] += 1
            distances[j] = total_variational_distance(degree_counts / np.sum(degree_counts), power_law_ba(max_steps))
            ba.update()
        distances_sum += distances
        if i == 0:
            plt.subplot(121)
            plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
            plt.subplot(122)
            plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
        else:
            plt.subplot(121)
            plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))
            plt.subplot(122)
            plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))

    plt.subplot(121)
    plt.plot(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
    plt.xlabel("Number of steps")
    plt.ylabel("Distance from power-law")
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.loglog(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
    plt.xlabel("Number of steps")
    plt.ylabel("Distance from power-law")
    plt.legend()
    plt.grid()

    plt.show()

def tbrw_convergence_plot(iterations = 50, max_steps = 500):
    gammas = [.5, .75, 1]
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
                plt.subplot(2, len(gammas), i+1)
                plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
                plt.subplot(2, len(gammas), i+1+len(gammas))
                plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1), label="Simulations")
            else:
                plt.subplot(2, len(gammas), i+1)
                plt.plot(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))
                plt.subplot(2, len(gammas), i+1+len(gammas))
                plt.loglog(np.arange(max_steps), distances, color=(0.5,0.5,1.0,0.1))

        plt.subplot(2, len(gammas), i+1)
        plt.plot(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
        plt.xlabel("Number of steps")
        plt.ylabel("Distance from power-law")
        plt.title(f"$\\gamma={round(gamma, 2)}$, linear")
        plt.legend()
        plt.grid()

        plt.subplot(2, len(gammas), i+1+len(gammas))
        plt.loglog(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
        plt.xlabel("Number of steps")
        plt.ylabel("Distance from power-law")
        plt.title(f"$\\gamma={round(gamma, 2)}$, loglog")
        plt.legend()
        plt.grid()

    plt.show()