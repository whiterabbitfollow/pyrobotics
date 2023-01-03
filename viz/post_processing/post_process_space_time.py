from examples.space_time.agent_and_adv.agent_n_adversary_world import AgentAdversary2DWorld
from pyrb.mp.planners.local_planners import LocalPlannerSpaceTimeInftyNorm
from pyrb.mp.post_processing.post_processing import PathPostProcessorContinuousSpaceTime, LOGGER_NAME_POST_PROCESSING
# from pyrb.mp.post_processing.post_processing import space_time_compute_path_cost
from pyrb.mp.utils.constants import LocalPlannerStatus
from pyrb.mp.utils.goal_regions import RealVectorMinimizingTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace

import numpy as np
import sys

import matplotlib.pyplot as plt

import logging

# logger = logging.getLogger(LOGGER_NAME_POST_PROCESSING)
# handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(handler)
#
# handler.setLevel(logging.DEBUG)
# logger.setLevel(logging.DEBUG)

TIME_HORIZON = 300

world = AgentAdversary2DWorld()
goal_region = RealVectorMinimizingTimeGoalRegion()
state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)
world.reset(seed=7)

local_planner = LocalPlannerSpaceTimeInftyNorm(
    min_coll_step_size=0.05,
    max_distance=(1.0, 20),
    max_actuation=world.robot.max_actuation
)

path = np.load("path.npy")
# print(np.cumsum(path[:, -1]))

path_processor = PathPostProcessorContinuousSpaceTime(
    space=state_space_start,
    local_planner=local_planner,
    goal_region=goal_region
)

path_pp = path_processor.post_process(path, max_cnt=100)
print(path_pp[1:] - path_pp[:-1])

# print(path[-1, -1])
# print(path_pp[-1, -1])
# plt.figure(1)
# plt.plot(path[:, 0], path[:, 1])
# plt.plot(path_pp[:, 0], path_pp[:, 1], marker=".")
# plt.xlim(world.robot.joint_limits[0, 0], world.robot.joint_limits[0, 1])
# plt.ylim(world.robot.joint_limits[1, 0], world.robot.joint_limits[1, 1])
# plt.show()