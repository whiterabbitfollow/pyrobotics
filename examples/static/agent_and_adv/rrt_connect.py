import matplotlib.pyplot as plt
import numpy as np

from examples.static.agent_and_adv.agent_n_adversary_world import AgentAdversary2DStaticWorld
from examples.utils import render_rrt_connect_planner_results
from pyrb.mp.planners.local_planners import LocalPlanner
from pyrb.mp.post_processing import post_process_path_continuous
from pyrb.mp.problem import PlanningProblem
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

path, status = problem.solve(state_start, goal_region, min_planning_time=0, max_planning_time=10)
path_pp = post_process_path_continuous(
        planner.space,
        planner.local_planner,
        path,
        goal_region,
        max_cnt=10,
        max_cnt_no_improvement=0
)

world.robot.set_config(state_start)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
render_rrt_connect_planner_results(ax1, ax2, world, planner, path, state_start, state_goal, goal_region)
ax2.plot(path_pp[:, 0], path_pp[:, 1], color="cyan", label="post processed")
plt.show()