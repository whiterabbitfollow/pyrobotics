from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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