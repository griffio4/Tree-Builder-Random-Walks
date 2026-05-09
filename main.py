from randomgraphs import TBRW, BA
import numpy as np
import matplotlib.pyplot as plt

leaf_pmf = lambda n: np.array([.9, .1])
tbrw = TBRW(leaf_pmf)

tbrw.update_to(2000)
tbrw.draw_fancy()

plt.show()
