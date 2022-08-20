import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib

import numpy as np

from examples.static.static_world import StaticBoxesWorld
from examples.utils import render_tree
from pyrb.mp.planners.static.rrt_star import RRTStarPlanner, RRTStarPlannerModified, logger

import logging
import sys

logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


matplotlib.rc("font", size=16)

world = StaticBoxesWorld()

np.random.seed(1)
world.reset()

planner = RRTStarPlanner(world, max_nr_vertices=int(1e4), nearest_radius=0.4, min_step_size_local_planner=0.2)
q_start = planner.sample_collision_free_config()
q_goal = planner.sample_collision_free_config()

path, status = planner.plan(q_start, q_goal, max_planning_time=10)
print(status.time_taken)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

world.robot.set_config(q_start)
world.render_world(ax1)
world.render_configuration_space(ax2)
vertices, edges = planner.tree.get_vertices(), planner.tree.get_edges()
render_tree(ax2, vertices, edges)

ax2.scatter(q_start[0], q_start[1], color="green", label="start, $q_I$")
ax2.scatter(q_goal[0], q_goal[1], color="red", label="goal, $q_G$")

if path.size:
    path = np.array(path)
    ax2.plot(path[:, 0], path[:, 1], color="blue", label="path")
    ax2.scatter(path[:, 0], path[:, 1], color="blue")

ax2.add_patch(Circle(q_goal, radius=planner.goal_region_radius, alpha=0.2, color="red"))
ax2.add_patch(Circle(q_start, radius=0.04, color="green"))

ax2.legend(loc="best")
ax2.set_title("Sampling based motion planning")
ax2.set_xlabel(r"$\theta_1$")
ax2.set_ylabel(r"$\theta_2$")

plt.show()
