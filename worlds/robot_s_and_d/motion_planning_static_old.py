import matplotlib.pyplot as plt

from functools import partial
import time

import pyrb
from pyrb.mp.planners.local_planners import LocalPlannerSpaceTime, LocalPlanner
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.problem import PlanningProblem
from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion, RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorStateSpace
from pyrb.mp.utils.trees.tree import Tree
from pyrb.rendering.utils import render_robot_configuration_meshes
from pyrb.traj.interpolate import interpolate_path, interpolate_along_line
from worlds.robot_s_and_d.render import render_obstacle_manager_meshes, render_trajectory_in_world, render_mesh
from worlds.robot_s_and_d.world import RobotCylinderWorld

import numpy as np

import tqdm

import trimesh


world = RobotCylinderWorld()
# file_name = "world.pckl"
np.random.seed(0)
world.reset()


limits = world.robot.joint_limits

thetas = np.linspace(limits[:, 0], limits[:, 1], 20).T
theta1, theta2, theta3 = thetas

w, h, d = np.diff(thetas).T[0]


T1, T2, T3 = np.meshgrid(theta1, theta2, theta3)
configs = np.hstack([T1.reshape(-1, 1), T2.reshape(-1, 1), T3.reshape(-1, 1)])

time_s = time.time()

meshes = []

for config in tqdm.tqdm(configs):
    world.robot.set_config(config)
    collision, objs_in_collision = world.robot.collision_manager.in_collision_other(
        world.obstacles.collision_manager, return_names=True
    )
    if collision:
        dynamic_collision = any("dynamic" in n for _, n in objs_in_collision)
        static_collision = any("static" in n for _, n in objs_in_collision)
        # config_colls.append(config)
        box = trimesh.creation.box(extents=(w, h, d))
        # objs_in_collision
        T = pyrb.kin.rot_trans_to_SE3(p=config)
        box.apply_transform(T)

        meshes.append((static_collision, dynamic_collision, box))

# config_colls = np.vstack(config_colls)
# print(config_colls.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ax.scatter(config_colls[:, 0], config_colls[:, 1], config_colls[:, 2], c="r")


for static_collision, dynamic_collision, box in meshes:
    if static_collision and dynamic_collision:
        color = "red"
    elif static_collision:
        color = "green"
    elif dynamic_collision:
        color = "blue"
    else:
        color = "black"
    render_mesh(ax, box, color=color)
ax.set_xlim(limits[0, 0], limits[0, 1])
ax.set_ylim(limits[1, 0], limits[1, 1])
ax.set_zlim(limits[2, 0], limits[2, 1])
plt.show()



# collision_mask = []
# for theta in thetas:
# self.robot.set_config(theta)
# collision = self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)
# collision_mask.append(not collision)
# collision_mask = np.array(collision_mask).reshape(resolution, resolution)
# ax.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)
# if self.start_config is not None:
# ax.scatter(self.start_config[0], self.start_config[1], label="Config", s=100)
# if self.robot.goal_state is not None:
# ax.scatter(self.robot.goal_state[0], self.robot.goal_state[1], s=100)
# ax.add_patch(Circle(tuple(self.robot.goal_state), radius=0.1, color="red", alpha=0.1, label="Goal set"))
# if path is not None:
# ax.plot(path[:, 0], path[:, 1], ls="-", marker=".", label="path")
# ax.set_title(r"Configuration space $\mathcal{C} \in \mathbb{R}^2$")
# ax.set_xlabel(r"$\theta_1$ [rad]")
# ax.set_ylabel(r"$\theta_2$ [rad]")
# ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
# ax.set_ylim(joint_limits[1, 0], joint_limits[1, 1])
# ax.set_aspect("equal")
# ax.legend(loc="best")







# world.save_state(file_name)

# world.load_state(file_name)
#
# for o in world.obstacles.dynamic_obstacles:
#     # print(world.obstacles)
#     print(o.positions.shape)

#
# for t in range(1000):
#     world.set_time(t)
#     collision, names = world.obstacles.collision_manager.in_collision_internal(return_names=True)
#     dynamic_coll = [(n1, n2) for n1, n2 in names if "dynamic" in n1 or "dynamic" in n2]
#     if collision and dynamic_coll:
#         print(dynamic_coll)

# PLANNING_TIME = 10
# TIME_HORIZON = 10
#
# #
# # 1.04719755
#
# # goal_region = RealVectorMinimizingTimeGoalRegion()
#
# state_space = RealVectorStateSpace(
#     world,
#     world.robot.nr_joints,
#     world.robot.joint_limits
# )
#
# goal_region = RealVectorGoalRegion()
# planner = RRTConnectPlanner(
#     space=state_space,
#     tree_start=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
#     tree_goal=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
#     local_planner=LocalPlanner(
#         min_coll_step_size=0.01,
#         max_distance=0.5
#     )
# )
#
# problem = PlanningProblem(
#     planner=planner
# )


# collision, names = world.obstacles.collision_manager.in_collision_internal(return_names=True)
# print([(n1, n2) for n1, n2 in names if "dynamic" in n1 or "dynamic" in n2])

#
#
# # world.set_time(0)
# # state_start = state_space.sample_feasible_state()
# #
# # render_robot_configuration_meshes(ax, world.robot, color="blue", alpha=0.5)
# # render_obstacle_manager_meshes(ax, world.obstacles, 0)
# # plt.show()
#
#
# while True:
#     state_start = state_space.sample_feasible_state()
#     world.robot.set_config(state_start)
#     collision, names = world.robot.collision_manager.in_collision_other(
#         world.obstacles.collision_manager,
#         return_names=True
#     )
#     if collision:
#         break
#
# print(collision)
# # names = None
# names = [n for _, n in names]

# plt.show()

# collision_cnt = 0
# success_cnt = 0
# cnt = 0
#
#
# result_str = ""

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# for i in range(10):
#     ts = np.random.randint(0, 100)
#     world.set_time(ts)
#     config = state_space.sample_collision_free_state()
#     config_goal = state_space.sample_collision_free_state()
#     goal_region.set_goal_state(config_goal)
#     sparse_path, status = problem.solve(
#         config, goal_region, min_planning_time=0, max_planning_time=np.inf
#     )
#     # print(sparse_path)
#     print(status.meta_data_problem)

#
# tbar = tqdm.tqdm()
#
# while True:
#     ts = np.random.randint(0, 100)
#     world.set_time(ts)
#     config = state_space.sample_collision_free_state()
#     config_goal = state_space.sample_collision_free_state()
#     goal_region.set_goal_state(config_goal)
#
#     collision = False
#     in_goal = False
#     nr_steps = 0
#
#     while not collision and not in_goal:
#         sparse_path, status = problem.solve(
#             config, goal_region, min_planning_time=0, max_planning_time=np.inf
#         )
#         if sparse_path.size > 0:
#             path = interpolate_path(sparse_path, partial(interpolate_along_line, max_distance_per_step=0.1))
#             config_next = path[1, :]
#         else:
#             config_next = config
#         ts += 1
#         world.set_time(ts)
#         collision_free = world.is_collision_free_state(config_next)
#         if collision_free:
#             config = config_next
#             in_goal = goal_region.is_within(config)
#         nr_steps += 1
#         tbar.set_description(
#             f"{result_str} distance to goal {np.linalg.norm(config-config_goal):.2f} nr steps: {nr_steps}"
#         )
#
#         # world.robot.set_config(config_goal)
#         # render_robot_configuration_meshes(ax, world.robot, color="orange", alpha=0.5)
#         #
#         # world.robot.set_config(config)
#         # render_robot_configuration_meshes(ax, world.robot, color="blue", alpha=0.5)
#         # render_obstacle_manager_meshes(ax, world.obstacles, ts)
#         # plt.show()
#
#
#     collision_cnt += collision
#     cnt += 1
#     success_cnt += in_goal
#     result_str = " ".join(
#         [
#             "%s: %f.2"
#             for s, v in zip(
#             ["cnt", "success_rate", "coll_rate"],
#             [cnt, success_cnt / cnt, collision_cnt / cnt]
#         )
#         ]
#     )
#     tbar.set_description(result_str)
#
# # trajectory = np.hstack([np.zeros((path.shape[0], 1)), path])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# while True:
#     render_trajectory_in_world(ax, world, trajectory)
#
