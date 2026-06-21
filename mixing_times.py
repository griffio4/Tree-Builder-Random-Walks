from randomgraphs import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm, colors





def mixing_visualization(gamma = 1/2, tree_size = 500, step_counts = [10, 100, 500, 2000]):
    
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


def stationary_distance_visualization(gamma = 1/2, tree_size = 1000, steps = 2000):

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


def mixing_time_plot(gamma = 1/2, tree_size = 100, steps = 2000):
    max_distances = []
    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    pi_T = stationary_distribution(tbrw.T)
    transition_matrix = tbrw.transition_matrix()
    distributions = np.identity(tree_size)

    for i in range(steps):
        print(f"Simulating step {i} of the random walk       ", end="\r")
        distributions @= transition_matrix
        distances = np.array([total_variational_distance(distributions[i], pi_T) for i in tbrw.T.nodes])
        max_distances.append(np.max(distances))
    
    plt.subplot(121)
    plt.plot(np.arange(steps), max_distances)
    plt.xlabel("t")
    plt.ylabel("d(t)")
    plt.grid()

    plt.subplot(122)
    plt.semilogy(np.arange(steps), max_distances)
    plt.xlabel("t")
    plt.ylabel("d(t)")
    plt.grid()

    plt.subplots_adjust(hspace=.2)
    plt.show()


def mixing_constants(tbrw: TBRW, steps = 1000, samples = 5):

    alphas = np.zeros(samples)
    Cs = np.zeros(samples)
    pi_T = stationary_distribution(tbrw.T)
    transition_matrix = tbrw.transition_matrix()
    transition_matrix_power = np.linalg.matrix_power(transition_matrix, steps)
    distributions = np.identity(tbrw.T.number_of_nodes())

    for i in range(samples):
        distributions @= transition_matrix_power
        distances = np.array([total_variational_distance(distributions[i], pi_T) for i in tbrw.T.nodes])
        d = np.max(distances)
        distributions @= transition_matrix
        distances = np.array([total_variational_distance(distributions[i], pi_T) for i in tbrw.T.nodes])
        alphas[i] = np.max(distances) / d
        Cs[i] = d / alphas[i]**(i*samples)

    alpha = np.mean(alphas)
    C = np.min(Cs)
    return alpha, C


def mixing_time_comparison(gamma = 1/2, tree_size = 100, steps = 10000):
    max_distances = []
    tbrw = GammaTBRW(gamma)
    tbrw.update_to_size(tree_size, True)
    pi_T = stationary_distribution(tbrw.T)
    transition_matrix = tbrw.transition_matrix()
    distributions = np.identity(tree_size)

    for i in range(steps):
        print(f"Simulating step {i} of the random walk       ", end="\r")
        distributions @= transition_matrix
        distances = np.array([total_variational_distance(distributions[i], pi_T) for i in tbrw.T.nodes])
        max_distances.append(np.max(distances))
    
    alpha = mixing_constants(tbrw)[0]
    max_distances = np.array(max_distances)

    log_bound = np.log(max_distances) / np.log(alpha)
    diameter_bound = (2 * nx.diameter(tbrw.T) + 1) * tbrw.T.number_of_edges() * np.log(2*tbrw.T.number_of_edges() / max_distances)
    
    
    plt.subplot(121)
    plt.plot(max_distances[:2000], np.arange(steps)[:2000], label="Mixing time")
    plt.plot(max_distances[:2000], log_bound[:2000], label="$\\log\\varepsilon/\\log\\alpha$")
    # plt.plot(max_distances, diameter_bound, label="Bound from Corollary 2.5")
    plt.gca().invert_xaxis()
    plt.xlabel("$\\varepsilon$")
    plt.ylabel("$t_{mix}$")
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.loglog(max_distances, np.arange(steps), label="Mixing time")
    plt.loglog(max_distances, log_bound, label="$\\log\\varepsilon/\\log\\alpha$")
    plt.loglog(max_distances, diameter_bound, label="Bound from Corollary 2.5")
    plt.gca().invert_xaxis()
    plt.xlabel("$\\varepsilon$")
    plt.ylabel("$t_{mix}$")
    plt.legend()
    plt.grid()

    plt.show()


def tbrw_alphas():
    iterations = 100
    for gamma in [1/6, 1/3, 1/2, 2/3]:
        means = []
        for size in [100, 200, 300, 400, 500]:
            alphas = np.zeros(iterations)
            for i in range(iterations):
                print(f"Iteration {i}", end="\r")
                tbrw = GammaTBRW(gamma)
                tbrw.update_to_size(size)
                alphas[i] = mixing_constants(tbrw)[0]
            means.append(np.mean(alphas))
            print(f"gamma={round(gamma, 2)}, size={size}, alpha_mean={np.mean(alphas)}, alpha_var={np.var(alphas)}\nalphas={alphas}                       ")
        print(f"Mean alpha values for gamma={round(gamma, 2)}: \n{np.array(means)}")

