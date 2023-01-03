import tqdm
from matplotlib.patches import Rectangle

from examples.space_time.make import compile_all_planners, Planners
from examples.space_time.moving_world import MovingBox1DimWorld
from examples.utils import render_tree
from pyrb.mp.planners.local_planners import LocalPlannerSpaceTime
from pyrb.mp.planners.rrt import RRTPlanner
from pyrb.mp.problem import PlanningProblem
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion, RealVectorMinimizingTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace
from pyrb.mp.utils.trees.tree import Tree
from worlds.robot_s_and_d.world import RobotCylinderWorld

np.random.seed(14)  # Challenging, solvable with ~200 steps...

world = RobotCylinderWorld()

np.random.seed(0)


TIME_HORIZON = 600

goal_region = RealVectorMinimizingTimeGoalRegion()


state_space = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)


local_planner = LocalPlannerSpaceTime(
    # min_path_distance=0.2,
    min_coll_step_size=0.05,
    max_distance=(1.0, 20),
    max_actuation=world.robot.max_actuation
)

planner = RRTPlanner(
    space=state_space,
    tree=Tree(max_nr_vertices=int(1e5), vertex_dim=state_space.dim),
    local_planner=local_planner
)


success_cnt = 0
cnt = 0
time_elapsed_acc = 0
time_elapsed_max = 0
max_nr_nodes_expanded_success = 0
max_nr_nodes_expanded_all = 0

tbar = tqdm.tqdm()

while True:
    t_start = np.random.randint(0, int(TIME_HORIZON * 0.1))
    world.set_time(t_start)
    config = state_space.sample_collision_free_config()
    config_goal = state_space.sample_collision_free_config()

    state_start = np.append(config, t_start)
    state_goal = np.append(config_goal, t_start + TIME_HORIZON)
    goal_region.set_goal_state(state_goal)

    problem = PlanningProblem(planner)
    path_sparse, data = problem.solve(
        state_start,
        goal_region,
        min_planning_time=0,
        max_planning_time=60
    )
    cnt += 1
    success = path_sparse.size > 0
    success_cnt += success
    if success:
        time_elapsed_acc += data.meta_data_problem["time_elapsed"]
        time_elapsed_max = max(time_elapsed_max, data.meta_data_problem["time_elapsed"])
        max_nr_nodes_expanded_success = max(max_nr_nodes_expanded_success, data.meta_data_planner["nr_nodes"])
    max_nr_nodes_expanded_all = max(max_nr_nodes_expanded_all, data.meta_data_planner["nr_nodes"])
    tbar.update(1)
    tbar.set_description(
        f"cnt: {cnt:0.2f} "
        f"success_rate: {success_cnt / cnt:0.2f} "
        f"mean_te: {time_elapsed_acc / cnt:0.2f} "
        f"max_te: {time_elapsed_max:0.2f} "
        f"max_nr_nodes_expanded_success: {max_nr_nodes_expanded_success: 0.2f} "
        f"max_nr_nodes_expanded_all: {max_nr_nodes_expanded_all: 0.2f}"
    )
