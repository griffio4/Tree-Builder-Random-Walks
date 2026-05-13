# from randomgraphs import *
import numpy as np
import matplotlib.pyplot as plt
from random import random

def approx_growth_times_1(gamma: int, end: int):
    return np.array([((1-gamma)*n+2**(gamma-1))**(1/(1-gamma)) for n in range(1, end+1)])

def approx_growth_times_2(gamma: int, end: int):
    return np.array([((1-gamma)*n)**(1/(1-gamma)) for n in range(1, end+1)])

def approx_growth_times_harmonic(end: int):
    return np.array([np.exp(n) for n in range(1, end+1)])

# simulate only the part of the TBRW that matters for the computation of growth times
def growth_times(max_length: int, gamma: float, log=False):
    n = 0
    times = []
    while True:
        n += 1
        if random() < n**(-gamma):
            times.append(n)
            if log:
                print(f"Growth time {len(times)}, n={n}        ", end="\r")
            if len(times) >= max_length:
                if log:
                    print("")
                return times

# Code used for first figure in section "Expected growth times"
def plot_growth_times():

    max_length = 100
    iterations = 250

    for i in range(1, 5):
        plt.subplot(2,2,i)
        total_times = np.zeros(max_length)
        gamma = i/6
        for j in range(iterations):
            print(f"Iteration {j+1}, gamma={i}/6        ", end="\r")
            times = growth_times(max_length, gamma)
            total_times += times
            if j == 0:
                plt.plot(times, color=(0.5,0.5,1.0,0.1), label="Simulations")
            else:
                plt.plot(times, color=(0.5,0.5,1.0,0.1))
        plt.plot(total_times / iterations, color="green", label="Empirical mean")
        plt.plot(approx_growth_times_1(gamma, max_length), color="red", label="$\\sqrt[1-\\gamma]{(1-\\gamma)n+2^{\\gamma-1}}$")
        plt.plot(approx_growth_times_2(gamma, max_length), color="orange", label="$\\sqrt[1-\\gamma]{(1-\\gamma)n}$")
        plt.xlabel("Number of growth times")
        plt.ylabel("Time steps")
        plt.title(f"$\\gamma={i}/6$")
        plt.legend()
        plt.grid()
    plt.suptitle(f"Empirical growth times compared to expected growth time, {iterations} simulations")
    plt.show()


# Code used for second figure in section "Expected growth times"
def plot_growth_times_harmonic():

    max_length = 10
    iterations = 250

    total_times = np.zeros(max_length)
    log_total_times = np.ones(max_length)
    for j in range(iterations):
        print(f"Iteration {j+1}")
        times = growth_times(max_length, 1, True)
        total_times += times
        log_total_times += np.log(times)
        if j == 0:
            plt.semilogy(times, color=(0.5,0.5,1.0,0.1), label="Simulations")
        else:
            plt.semilogy(times, color=(0.5,0.5,1.0,0.1))
    plt.semilogy(total_times / iterations, color="green", label="Empirical mean")
    plt.semilogy(np.exp(log_total_times / iterations), color="purple", label="Empirical geometric mean")
    plt.semilogy(approx_growth_times_harmonic(max_length), color="red", label="exp$(n)$")
    plt.xlabel("Number of growth times")
    plt.ylabel("Time steps")
    plt.legend()
    plt.grid()
    plt.title(f"Empirical growth times compared to expected growth time, {iterations} simulations")
    plt.show()

plot_growth_times()
plot_growth_times_harmonic()
