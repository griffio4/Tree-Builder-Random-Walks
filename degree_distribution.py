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

def tbrw_convergence_plot(iterations = 50, max_steps = 3000, gamma=1/2):
    distances_sum = np.zeros(max_steps)
    for i in range(iterations):
        distances = np.zeros(max_steps)
        tbrw = GammaTBRW(gamma)
        for j in range(max_steps):
            print(f"Iteration {i}, step {j}          ", end="\r")
            deg = degrees(tbrw.T)
            degree_counts = np.zeros(max_steps)
            for d in deg:
                degree_counts[d-1] += 1
            distances[j] = total_variational_distance(degree_counts / np.sum(degree_counts), power_law_ba(max_steps))
            tbrw.update_to_size(tbrw.T.number_of_nodes() + 1)
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
    plt.xlabel("Tree size")
    plt.ylabel("Distance from power-law")
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.loglog(np.arange(max_steps), distances_sum / iterations, color="red", label="Mean of simulations")
    plt.xlabel("Tree size")
    plt.ylabel("Distance from power-law")
    plt.legend()
    plt.grid()

    plt.show()

def tbrw_convergence_small_gamma(iterations = 50, max_steps = 3000, gammas=[0, .1, .2, .3, .4, .5]):
    for gamma in gammas:
        distances_sum = np.zeros(max_steps)
        for i in range(iterations):
            distances = np.zeros(max_steps)
            tbrw = GammaTBRW(gamma)
            for j in range(max_steps):
                print(f"gamma={gamma}, iteration {i}, step {j}          ", end="\r")
                deg = degrees(tbrw.T)
                degree_counts = np.zeros(max_steps)
                for d in deg:
                    degree_counts[d-1] += 1
                distances[j] = total_variational_distance(degree_counts / np.sum(degree_counts), power_law_ba(max_steps))
                tbrw.update_to_size(tbrw.T.number_of_nodes() + 1)
            distances_sum += distances

        plt.subplot(121)
        plt.plot(np.arange(max_steps), distances_sum / iterations, label=f"$\\gamma={round(gamma, 2)}$")
        plt.xlabel("Tree size")
        plt.ylabel("Distance from power-law")
        plt.legend()

        plt.subplot(122)
        plt.loglog(np.arange(max_steps), distances_sum / iterations, label=f"$\\gamma={round(gamma, 2)}$")
        plt.xlabel("Tree size")
        plt.ylabel("Distance from power-law")
        plt.legend()
    
    plt.subplot(121)
    plt.grid()
    plt.subplot(122)
    plt.grid()
    plt.show()

tbrw_convergence_small_gamma()