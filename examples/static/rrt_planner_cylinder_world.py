from matplotlib.patches import Circle

from examples.static.cylinder_world import CylinderWorld
from examples.static.static_world import StaticBoxesWorld
from examples.utils import plot_rrt_planner_results, render_tree
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt import RRTPlanner
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorStateSpace
from pyrb.mp.utils.trees.tree import Tree

from pyrb.mp.planners.local_planners import LocalPlanner

import numpy as np

from pyrb.mp.utils.trees.tree_rewire import TreeRewire

world = CylinderWorld()
world.reset()

state_space = RealVectorStateSpace(
    world,
    world.robot.state_dim,
    world.robot.joint_limits
)

goal_region = RealVectorGoalRegion()

local_planner = LocalPlanner(
    min_coll_step_size=0.05,
    max_distance=0.5,
    include_distance_to_obst=True
)


# planner = RRTConnectPlanner(
#     space=state_space,
#     tree_start=TreeRewire(
#         local_planner=local_planner,
#         space=state_space,
#         nearest_radius=1.0,
#         max_nr_vertices=int(1e4)
#     ),
#     tree_goal=TreeRewire(
#         local_planner=local_planner,
#         space=state_space,
#         nearest_radius=1.0,
#         max_nr_vertices=int(1e4)
#     ),
#     local_planner=local_planner
# )
# 1731

planner = RRTPlanner(
    space=state_space,
    tree=TreeRewire(
        local_planner=local_planner,
        space=state_space,
        nearest_radius=0.3,
        max_nr_vertices=int(1e4)
    ),
    local_planner=local_planner
)

problem = PlanningProblem(
    planner=planner
)

state_start = np.array([-.5, 0])
state_goal = np.array([.5, 0])


goal_region.set_goal_state(state_goal)

path, status = problem.solve(
    state_start, goal_region, min_planning_time=10, max_planning_time=20
)


import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

world.robot.set_config(state_start)
world.render_world(ax1)

tree = planner.tree

print(tree.vert_cnt)

vertices, edges = tree.get_vertices(), tree.get_edges()
render_tree(ax1, vertices, edges)

ax1.scatter(state_start[0], state_start[1], color="green", label="start, $q_I$")
ax1.scatter(state_goal[0], state_goal[1], color="red", label="goal, $q_G$")

if path.size > 0:
    ax1.plot(path[:, 0], path[:, 1], color="orange", label="path", ls="-", marker=".")

ax1.add_patch(Circle(state_goal, radius=goal_region.radius, alpha=0.1, color="red"))
ax1.add_patch(Circle(state_start, radius=1.0, color="green", alpha=.1))

ax1.legend(loc="best")
ax1.set_title("Sampling based motion planning")
ax1.set_xlabel(r"$\theta_1$")
ax1.set_ylabel(r"$\theta_2$")
plt.show()
