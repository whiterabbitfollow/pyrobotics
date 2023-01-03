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

np.random.seed(0)
world.reset()


state_space = RealVectorStateSpace(
    world,
    world.robot.nr_joints,
    world.robot.joint_limits
)

goal_region = RealVectorGoalRegion()
planner = RRTConnectPlanner(
    space=state_space,
    tree_start=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
    tree_goal=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
    local_planner=LocalPlanner(
        min_coll_step_size=0.01,
        max_distance=0.5
    )
)

problem = PlanningProblem(
    planner=planner
)

collision_cnt = 0
success_cnt = 0
terminated_cnt = 0
cnt = 0


result_str = ""
tbar = tqdm.tqdm()

while True:
    ts = np.random.randint(0, 100)
    world.set_time(ts)
    config = state_space.sample_collision_free_state()
    config_goal = state_space.sample_collision_free_state()
    goal_region.set_goal_state(config_goal)

    collision_free = True
    in_goal = False
    nr_steps = 0
    max_nr_steps_still = 50
    nr_steps_still = 0
    terminate = False

    while collision_free and not in_goal and not terminate:
        sparse_path, status = problem.solve(
            config, goal_region, min_planning_time=0, max_planning_time=5
        )
        if sparse_path.size > 0:
            path = interpolate_path(sparse_path, partial(interpolate_along_line, max_distance_per_step=0.1))
            config_next = path[1, :]
        else:
            config_next = config
            nr_steps_still += 1
        ts += 1
        world.set_time(ts)
        collision_free = world.is_collision_free_state(config_next)
        if collision_free:
            config = config_next
            in_goal = goal_region.is_within(config)
            world.robot.set_config(config)

        nr_steps += 1
        tbar.set_description(
            f"{result_str} distance to goal "
            f"{np.linalg.norm(config-config_goal):.2f} "
            f"nr steps: {nr_steps} "
            f"standing still steps: {nr_steps_still}"
        )
        terminate = nr_steps_still > max_nr_steps_still
        # debug I guess
        # world.robot.set_config(config_goal)
        # render_robot_configuration_meshes(ax, world.robot, color="orange", alpha=0.5)
        # render_robot_configuration_meshes(ax, world.robot, color="blue", alpha=0.5)
        # render_obstacle_manager_meshes(ax, world.obstacles, ts)
        # plt.show()

    collision_cnt += not collision_free
    terminated_cnt += terminate
    cnt += 1
    success_cnt += in_goal
    result_str = " ".join(
        [
            "%s: %f.2" % (s, v)
            for s, v in zip(
            ["cnt", "success_rate", "coll_rate", "terminated_ratio"],
            [cnt, success_cnt / cnt, collision_cnt / cnt, terminated_cnt/cnt]
        )
        ]
    )
    tbar.set_description(result_str)



#
# # trajectory = np.hstack([np.zeros((path.shape[0], 1)), path])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# while True:
#     render_trajectory_in_world(ax, world, trajectory)
#
