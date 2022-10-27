import logging
import sys

import numpy as np
import matplotlib.pyplot as plt


from examples.static.static_world import StaticBoxesWorld
from examples.utils import render_rrt_planner_results
from pyrb.mp.problem import PlanningProblem
from pyrb.mp.planners.rrt import RRT_PLANNER_LOGGER_NAME
from pyrb.mp.planners.local_planners import LocalPlanner


from pyrb.mp.planners.rrt_informed import RRTInformedPlanner
from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorStateSpace
from pyrb.mp.utils.trees.tree_rewire import TreeRewire

logger = logging.getLogger(RRT_PLANNER_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

np.random.seed(4)

world = StaticBoxesWorld()
world.reset()

state_space = RealVectorStateSpace(
    world,
    world.robot.nr_joints,
    world.robot.joint_limits
)

goal_region = RealVectorGoalRegion()

local_planner = LocalPlanner(
    min_coll_step_size=0.1,
    max_distance=0.5
)

planner = RRTInformedPlanner(
    space=state_space,
    tree=TreeRewire(
        local_planner=local_planner,
        space=state_space,
        nearest_radius=0.5,
        max_nr_vertices=int(1e4)
    ),
    local_planner=local_planner
)

problem = PlanningProblem(
    planner=planner
)


def draw_unit_ball(planner):
    CL, x_center = planner.CL, planner.x_center
    thetas = np.linspace(0, np.pi * 2, 100)
    xs = np.vstack([np.cos(thetas), np.sin(thetas)])
    x_center = x_center.reshape(-1, 1)
    xs = CL @ xs + x_center
    return xs.T


state_start = state_space.sample_collision_free_state()
state_goal = state_space.sample_collision_free_state()
goal_region.set_goal_state(state_goal)


path, data = problem.solve(
    state_start, goal_region, min_planning_time=np.inf, max_iters=200
)


print(data.meta_data_problem["iter_cnt"])



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

render_rrt_planner_results(
    ax1, ax2, world, planner, path, state_start, state_goal, goal_region
)

if path.size:
    vals = draw_unit_ball(planner)
    ax2.plot(vals[:, 0], vals[:, 1])
    ax2.set_aspect("equal")

plt.show()



