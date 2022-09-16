import matplotlib.pyplot as pl
import matplotlib
import numpy as np

from examples.static.agent_and_adv.agent_n_adversary_world import AgentAdversary2DStaticWorld
from examples.utils import plot_rrt_connect_planner_results
from pyrb.mp.planners.local_planners import LocalPlanner
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorStateSpace
from pyrb.mp.utils.trees.tree import Tree


np.random.seed(4)

world = AgentAdversary2DStaticWorld()

world.reset()
world.reset_config()

config = world.robot.config
t = 0
world.robot.set_config(config)
world.set_static_time(t)

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
        min_coll_step_size=0.05,
        max_distance=0.5
    )
)

problem = PlanningProblem(
    planner=planner
)

world.reset_config()

state_start = world.start_config.copy()
state_goal = world.robot.goal_state.copy()

goal_region.set_goal_state(state_goal)

path, status = problem.solve(state_start, goal_region, min_planning_time=0, max_planning_time=1)




print(path)




# plot_rrt_connect_planner_results(
#     world, planner, path, state_start, state_goal, goal_region
# )
