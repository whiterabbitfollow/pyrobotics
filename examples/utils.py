from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

from pyrb.rendering.utils import robot_configuration_to_matplotlib_rectangles


def render_manipulator_on_axis(ax, manipulator, color, alpha=1.0):
    rectangles = []
    rectangles_params = robot_configuration_to_matplotlib_rectangles(manipulator)
    for rectangles_param in rectangles_params:
        rectangles.append(Rectangle(*rectangles_param))
    coll = PatchCollection(rectangles)
    coll.set_color(color)
    coll.set_alpha(alpha)
    ax.add_collection(coll)


def render_tree(ax, vertices, edges, color="black"):
    ax.scatter(vertices[:, 0], vertices[:, 1], color=color)
    for i_child, i_parent in enumerate(edges):
        if i_parent >= vertices.shape[0]:
            ax.scatter(vertices[i_child, 0], vertices[i_child, 1], color="red", marker="*")
        else:
            q = np.stack([vertices[i_parent], vertices[i_child]], axis=0)
            ax.plot(q[:, 0], q[:, 1], color=color)

