import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib

import numpy as np

from examples.static.static_world import StaticBoxesWorld
from pyrb.mp.planners.static.rrt_star_connect import RRTStarConnectPlanner


matplotlib.rc("font", size=16)

np.random.seed(22)
world = StaticBoxesWorld()
world.reset()

planner = RRTStarConnectPlanner(
    world,
    max_nr_vertices=int(1e3)
)

q_start = planner.sample_collision_free_config()
q_goal = planner.sample_collision_free_config()

path, status = planner.plan(q_start, q_goal, min_planning_time=1, max_planning_time=10)
print(status.time_taken)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
world.render_world(ax1)
world.render_configuration_space(ax2)

for tree, color in ((planner.tree_start, "blue"), (planner.tree_goal, "red")):
    vert_cnt = tree.vert_cnt
    verts = tree.vertices
    ax2.scatter(verts[:vert_cnt, 0], verts[:vert_cnt, 1], color="black")
    for i_parent, indxs_children in tree.edges_parent_to_children.items():
        for i_child in indxs_children:
            q = np.stack([verts[i_parent], verts[i_child]], axis=0)
            ax2.plot(q[:, 0], q[:, 1], color=color)

ax2.scatter(q_start[0], q_start[1], color="green", label="start, $q_I$")
ax2.scatter(q_goal[0], q_goal[1], color="red", label="goal, $q_G$")


if path.size > 0:
    ax2.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")

ax2.add_patch(Circle(q_goal, radius=planner.goal_region_radius, alpha=0.2, color="red"))
ax2.add_patch(Circle(q_start, radius=0.04, color="green"))

ax2.legend(loc="best")
ax2.set_title("Sampling based motion planning")
ax2.set_xlabel(r"$\theta_1$")
ax2.set_ylabel(r"$\theta_2$")

plt.show()
