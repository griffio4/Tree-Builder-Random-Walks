from randomgraphs import *
import numpy as np
import matplotlib.pyplot as plt

for i in range(1,5):
    plt.subplot(2,2,i)
    tbrw = GammaTBRW(i/6)
    tbrw.update_to_size(250, True)
    tbrw.draw_fancy()
    plt.title(f"$\\gamma={round(i/6, 2)}$")
plt.show()

