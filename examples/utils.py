import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
import numpy as np

matplotlib.rc("font", size=16)


def render_tree(ax, vertices, edges, color="black", lw=1):
    ax.scatter(vertices[:, 0], vertices[:, 1], color=color)
    for i_child, i_parent in enumerate(edges):
        if i_parent >= vertices.shape[0]:
            ax.scatter(vertices[i_child, 0], vertices[i_child, 1], color="red", marker="*")
        else:
            q = np.stack([vertices[i_parent], vertices[i_child]], axis=0)
            ax.plot(q[:, 0], q[:, 1], color=color, lw=lw)


def plot_rrt_planner_results(world, planner, path, state_start, state_goal, goal_region):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    render_rrt_planner_results(ax1, ax2, world, planner, path, state_start, state_goal, goal_region)
    plt.show()


def render_rrt_planner_results(ax1, ax2, world, planner, path, state_start, state_goal, goal_region):
    world.robot.set_config(state_start)
    world.render_world(ax1)
    world.render_configuration_space(ax2)

    vertices, edges = planner.tree.get_vertices(), planner.tree.get_edges()
    render_tree(ax2, vertices, edges)

    ax2.scatter(state_start[0], state_start[1], color="green", label="start, $q_I$")
    ax2.scatter(state_goal[0], state_goal[1], color="red", label="goal, $q_G$")

    if path.size > 0:
        ax2.plot(path[:, 0], path[:, 1], color="orange", label="path", ls="-", marker=".")

    ax2.add_patch(Circle(state_goal, radius=goal_region.radius, alpha=0.2, color="red"))
    ax2.add_patch(Circle(state_start, radius=0.04, color="green"))

    ax2.legend(loc="best")
    ax2.set_title("Sampling based motion planning")
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")


def plot_rrt_connect_planner_results(world, planner, path, state_start, state_goal, goal_region):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    render_rrt_connect_planner_results(ax1, ax2, world, planner, path, state_start, state_goal, goal_region)
    plt.show()


def render_rrt_connect_planner_results(ax1, ax2, world, planner, path, state_start, state_goal, goal_region):

    world.render_world(ax1)
    world.render_configuration_space(ax2)

    ax2.scatter(state_start[0], state_start[1], color="green", label="start, $q_I$")
    ax2.scatter(state_goal[0], state_goal[1], color="red", label="goal, $q_G$")

    for tree, color in ((planner.tree_start, "blue"), (planner.tree_goal, "red")):
        vertices, edges = tree.get_vertices(), tree.get_edges()
        render_tree(ax2, vertices, edges, color=color)

    if path.size > 0:
        ax2.plot(path[:, 0], path[:, 1], color="orange", label="path", ls="-", marker=".")

    ax2.add_patch(Circle(state_goal, radius=goal_region.radius, alpha=0.2, color="red"))
    ax2.add_patch(Circle(state_start, radius=0.04, color="green"))

    ax2.legend(loc="best")
    ax2.set_title("Sampling based motion planning")
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")


