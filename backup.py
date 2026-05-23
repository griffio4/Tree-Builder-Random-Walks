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