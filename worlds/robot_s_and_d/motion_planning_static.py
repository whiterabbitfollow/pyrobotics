import matplotlib.pyplot as plt

from functools import partial
import time

import pyrb
from pyrb.mp.planners.local_planners import LocalPlannerSpaceTime, LocalPlanner
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.problem import PlanningProblem, SolutionStatus
from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion, RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorStateSpace
from pyrb.mp.utils.trees.tree import Tree
from worlds.robot_s_and_d.world import RobotCylinderWorld



import numpy as np
import tqdm


import trimesh


world = RobotCylinderWorld()
# file_name = "world.pckl"
np.random.seed(0)
tbar = tqdm.tqdm(range(1, 101))
success_cnt = 0

state_space = RealVectorStateSpace(
    world,
    world.robot.nr_joints,
    world.robot.joint_limits
)
goal_region = RealVectorGoalRegion()

time_elapsed_acc = 0
max_time_elapsed = 0

for cnt in tbar:
    world.reset()
    ts = np.random.randint(0, 100)
    world.set_time(ts)
    config = state_space.sample_collision_free_state()
    config_goal = state_space.sample_collision_free_state()
    goal_region.set_goal_state(config_goal)
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
    sparse_path, data = problem.solve(
        config, goal_region, min_planning_time=0, max_planning_time=10
    )
    success = data.status == SolutionStatus.SUCCESS
    if success:
        time_elapsed = data.meta_data_problem["time_elapsed"]
        time_elapsed_acc += time_elapsed
        max_time_elapsed = max(time_elapsed, max_time_elapsed)

    success_cnt += success
    tbar.set_description(
        f"Success-ratio: {success_cnt / cnt}, "
        f"Mean-time-elapsed: {time_elapsed_acc / success_cnt if success_cnt else np.nan} "
        f"Max-time-elapsed: {max_time_elapsed if max_time_elapsed else np.nan}"
    )
    # collision_cnt = 0
    # success_cnt = 0
    # cnt = 0
    #
    # ts = np.random.randint(0, 100)
    # world.set_time(ts)
    #
    # config = state_space.sample_collision_free_state()
    # config_goal = state_space.sample_collision_free_state()
    # goal_region.set_goal_state(config_goal)
    #
    # collision = False
    # in_goal = False
    # nr_steps = 0
    #
    # sparse_path, status = problem.solve(
    #     config, goal_region, min_planning_time=0, max_planning_time=10
    # )
    #
    # if sparse_path.size > 0:
    #     path = interpolate_path(sparse_path, partial(interpolate_along_line, max_distance_per_step=0.1))


# in_goal = goal_region.is_within(config)
# world.robot.set_config(config)

# print(sparse_path)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# render_obstacle_space(ax, world)
# ax.scatter(config[0], config[1], config[2], c="b")
# ax.scatter(config_goal[0], config_goal[1], config_goal[2], c="r")
# plt.show()
