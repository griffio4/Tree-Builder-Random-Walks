import growth_times
import mixing_times
import degree_distribution
import animations

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
degree_distribution.tbrw_convergence_plot()
degree_distribution.tbrw_convergence_small_gamma()
degree_distribution.tbrw_convergence_small_gamma(iterations=50, max_steps=100, growths_per_step=500, gammas=[0, .05, .1, .15, .2, .25, .3])

# animation used in presentation
animations.growth_animation()