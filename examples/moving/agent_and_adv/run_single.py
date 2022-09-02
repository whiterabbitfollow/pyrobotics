import numpy as np
import tqdm

from examples.moving.agent_and_adv.agent_n_adversary_world import AgentAdversary2DWorld
from examples.moving.make import compile_all_planners
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.utils.goal_regions import RealVectorMinimizingTimeGoalRegion, GOAL_REGIONS_LOGGER_NAME
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace

import logging
import sys


np.random.seed(9)
MAX_PLANNING_TIME = 100
MIN_PLANNING_TIME = 20

TIME_HORIZON = 300

world = AgentAdversary2DWorld()

goal_region = RealVectorMinimizingTimeGoalRegion()

state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)
state_space_goal = RealVectorPastTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

planners = compile_all_planners(world, state_space_start, state_space_goal)


for planner_name in ("rrt_star", "rrt"):
    planner = planners[planner_name]

    logger = logging.getLogger(GOAL_REGIONS_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    world.reset()
    state_start = np.append(world.robot.config, 0)
    goal_config = world.robot.goal_state
    state_goal = np.append(goal_config, TIME_HORIZON)
    goal_region.set_goal_state(state_goal)

    problem = PlanningProblem(planner)

    path, status = problem.solve(
        state_start,
        goal_region,
        max_planning_time=MAX_PLANNING_TIME,
        min_planning_time=MIN_PLANNING_TIME
    )
    print(path)
