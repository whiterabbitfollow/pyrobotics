import numpy as np

from examples.moving.agent_and_adv.agent_n_adversary_world import AgentAdversary2DWorld
from examples.moving.make import compile_all_planners, Planners
from pyrb.mp.post_processing import post_process_path_space_time_minimal_time, LOGGER_NAME_POST_PROCESSING
from pyrb.mp.problem import PlanningProblem
from pyrb.mp.utils.goal_regions import RealVectorMinimizingTimeGoalRegion, GOAL_REGIONS_LOGGER_NAME
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace

import logging
import sys


np.random.seed(11)


MIN_PLANNING_TIME = 10
MAX_PLANNING_TIME = 100


TIME_HORIZON = 300

world = AgentAdversary2DWorld()

goal_region = RealVectorMinimizingTimeGoalRegion()

state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

state_space_goal = RealVectorPastTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON
)

planners = compile_all_planners(world, state_space_start, state_space_goal)

logger = logging.getLogger(GOAL_REGIONS_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(LOGGER_NAME_POST_PROCESSING)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


planner = planners[Planners.RRT_CONNECT]

world.reset()
state_start = np.append(world.robot.config, 0)
goal_config = world.robot.goal_state
state_goal = np.append(goal_config, TIME_HORIZON)
goal_region.set_goal_state(state_goal)

problem = PlanningProblem(planner)

path, status = problem.solve(
    state_start,
    goal_region,
    min_planning_time=MIN_PLANNING_TIME,
    max_planning_time=MAX_PLANNING_TIME
)

print("Post processing path")
print("Time horizon before preprocessing", path[-1, -1])

path_pp = post_process_path_space_time_minimal_time(
    state_space_start,
    planner.local_planner,
    goal_region,
    path,
    max_cnt_no_improvement=10
)

print("Time horizon after preprocessing", path_pp[-1, -1])

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(path[:, 0], path[:, 1])
plt.plot(path_pp[:, 0], path_pp[:, 1])
plt.show()
