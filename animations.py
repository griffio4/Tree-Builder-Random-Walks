from randomgraphs import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

def growth_animation():

    tbrw = GammaTBRW(0.5)
    tbrw.update_to_size(500)
    pos = tbrw.get_positions()
    nodes = list(tbrw.T.nodes)

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()

        # Nodes visible up to current frame
        visible_nodes = nodes[:frame + 1]

        # Induced subgraph
        T = tbrw.T.subgraph(visible_nodes)

        nx.draw(
            T,
            pos,
            node_color=node_colors(T.number_of_nodes()),
            node_size=30
        )

        ax.set_title(f"Step {frame + 1}")

    def node_colors(count: int):
        color1 = (.4, .8, .8)
        color2 = (.8, .0, .0)

        def mix(color1, color2, t):
            return (t * color2[0] + (1-t) * color1[0], t * color2[1] + (1-t) * color1[1], t * color2[2] + (1-t) * color1[2])
        
        colors = [mix(color1, color2, (i/count)**3) for i in range(count)]
        return colors

    anim = FuncAnimation(
        fig,
        update,
        frames=len(nodes),
        interval=10,  # ms between frames
        repeat=False
    )

    anim.save(filename="growth_animation.gif", writer="pillow")
    plt.show()
