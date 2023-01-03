import matplotlib.pyplot as plt

from functools import partial
import time

import itertools

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

mask = np.zeros((20, 20, 20))

time_s = time.time()

meshes = []

# for i, config in enumerate(configs):
#     world.robot.set_config(config)
#     collision = world.robot.collision_manager.in_collision_other(world.obstacles.collision_manager)
#     if collision:
#         # config_colls.append(config)
#         box = trimesh.creation.box(extents=(w, h, d))
#         T = pyrb.kin.rot_trans_to_SE3(p=config)
#         box.apply_transform(T)
#         meshes.append(box)
#         mask[np.unravel_index(i, (20, 20, 20))] = 1
#         # np.ind

actions = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
])

coord = np.array([
    [0, 0, 0],
    [1, 0, 0]
])

# coord_neigh = np.array([]).reshape(coord.shape + (0,))

coord_neigh = np.concatenate([(coord + a)[None] for a in actions], axis=0)
mask_valid = np.all((0 <= coord_neigh) & (coord_neigh < 20), axis=2)
coord_neigh = coord_neigh[mask_valid]
mask[coord_neigh]


# coords_check = coord + actions
# mask_valid = np.all((0 <= coords_check) & (coords_check < 20), axis=1)
# coords_check = coords_check[mask_valid]
# print(coords_check)
# skip = mask[coords_check].all()
# config_colls = np.vstack(config_colls)
# print(config_colls.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(config_colls[:, 0], config_colls[:, 1], config_colls[:, 2], c="r")
# for m in meshes:
#     render_mesh(ax, m, color="red")
# ax.set_xlim(limits[0, 0], limits[0, 1])
# ax.set_ylim(limits[1, 0], limits[1, 1])
# ax.set_zlim(limits[2, 0], limits[2, 1])
# plt.show()