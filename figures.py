import growth_times
import mixing_times
import degree_distribution

# Collection of functions used to generate plots, tables and diagrams in the paper

# Growth time plots
growth_times.plot_growth_times()
growth_times.plot_growth_times_harmonic()

# Mixing visualizations
mixing_times.mixing_visualization()
mixing_times.stationary_distance_visualization()

# Mixing time plots
mixing_times.mixing_time_plot()
mixing_times.mixing_time_comparison()

# Mixing constants (table)
mixing_times.mixing_constants()

# Degree distribution convergence rate plots
degree_distribution.ba_convergence_plot()
