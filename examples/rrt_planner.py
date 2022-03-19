import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import matplotlib

import numpy as np

from pyrb.world import StaticBoxesWorld
from pyrb.planners.rrt import RRTPlanner


matplotlib.rc("font", size=16)

world = StaticBoxesWorld()
world.reset()
planner = RRTPlanner(world, max_nr_vertices=int(1e2))
q_start = planner.sample_collision_free_config()
q_goal = planner.sample_collision_free_config()
path = planner.plan(q_start, q_goal)


fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

joint_limits = world.robot.get_joint_limits()

thetas_1 = np.linspace(joint_limits[0, 0], joint_limits[0, 1], 100)
thetas_2 = np.linspace(joint_limits[1, 0], joint_limits[1, 1], 100)
theta_grid_1, theta_grid_2 = np.meshgrid(thetas_1, thetas_2)
thetas = np.stack([theta_grid_1.ravel(), theta_grid_2.ravel()], axis=1)

collision_mask = []

robot = world.robot

for theta in thetas:
    robot.set_config(theta)
    collision = robot.collision_manager.in_collision_other(world.obstacle_region.collision_manager)
    collision_mask.append(not collision)

collision_mask = np.array(collision_mask).reshape(100, 100)
ax1.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)

ax1.scatter(planner.vertices[:planner.vert_cnt, 0], planner.vertices[:planner.vert_cnt, 1], color="black", s=10)

for i_parent, indxs_children in planner.edges_parent_to_children.items():
    for i_child in indxs_children:
        q = np.stack([planner.vertices[i_parent], planner.vertices[i_child]], axis=0)
        ax1.plot(q[:, 0], q[:, 1], color="black")

ax1.scatter(q_start[0], q_start[1], color="green", label="start, $q_I$")
ax1.scatter(q_goal[0], q_goal[1], color="red", label="goal, $q_G$")

if path:
    path = np.array(path)
    ax1.plot(path[:, 0], path[:, 1], color="blue", label="path")
    ax1.scatter(path[:, 0], path[:, 1], color="blue")

ax1.add_patch(Circle(q_goal, radius=planner.goal_region_radius, alpha=0.2, color="red"))
ax1.add_patch(Circle(q_start, radius=0.04, color="green"))

ax1.legend(loc="best")
ax1.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
ax1.set_ylim(joint_limits[1, 0], joint_limits[1, 1])

ax1.set_title("Sampling based motion planning")
ax1.set_xlabel(r"$\theta_1$")
ax1.set_ylabel(r"$\theta_2$")
plt.show()



