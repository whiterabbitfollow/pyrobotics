from examples.static.static_world import StaticBoxesWorld
from examples.utils import plot_rrt_planner_results
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt import RRTPlanner, RRT_PLANNER_LOGGER_NAME, LocalPlanner

import logging
import sys

from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorStateSpace
from pyrb.mp.utils.trees.tree_rewire import TreeRewire

logger = logging.getLogger(RRT_PLANNER_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

world = StaticBoxesWorld()
world.reset()

state_space = RealVectorStateSpace(
    world,
    world.robot.nr_joints,
    world.robot.joint_limits
)

goal_region = RealVectorGoalRegion()

local_planner = LocalPlanner(
    min_path_distance=0.1,
    min_coll_step_size=0.1,
    max_distance=0.5
)

planner = RRTPlanner(
    space=state_space,
    tree=TreeRewire(
        local_planner=local_planner,
        space=state_space,
        nearest_radius=1.0,
        max_nr_vertices=int(1e4)
    )
)

problem = PlanningProblem(
    planner=planner
)

state_start = state_space.sample_collision_free_state()
state_goal = state_space.sample_collision_free_state()
goal_region.set_goal_state(state_goal)

path, status = problem.solve(
    state_start, goal_region, min_planning_time=1, max_planning_time=10
)

plot_rrt_planner_results(
    world, planner, path, state_start, state_goal, goal_region
)
