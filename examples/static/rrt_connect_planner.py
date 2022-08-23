import numpy as np

from examples.static.static_world import StaticBoxesWorld
from examples.utils import plot_rrt_connect_planner_results
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt import LocalPlanner
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorStateSpace
from pyrb.mp.utils.trees.tree import Tree

np.random.seed(22)
world = StaticBoxesWorld()
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
        min_path_distance=0.1,
        min_coll_step_size=0.05,
        max_distance=0.5
    )
)

problem = PlanningProblem(
    planner=planner
)


state_start = state_space.sample_collision_free_state()
state_goal = state_space.sample_collision_free_state()
goal_region.set_goal_state(state_goal)

path, status = problem.solve(state_start, goal_region, min_planning_time=1, max_planning_time=10)

plot_rrt_connect_planner_results(
    world, planner, path, state_start, state_goal, goal_region
)
