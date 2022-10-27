from matplotlib.patches import Rectangle

from examples.space_time.make import compile_all_planners, Planners
from examples.space_time.moving_world import MovingBox1DimWorld
from examples.utils import render_tree
from pyrb.mp.problem import PlanningProblem
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace


np.random.seed(14)  # Challenging, solvable with ~200 steps...

world = MovingBox1DimWorld()
world.reset()


PLANNING_TIME = 10
TIME_HORIZON = 60

#
# 1.04719755

# goal_region = RealVectorMinimizingTimeGoalRegion()
goal_region = RealVectorTimeGoalRegion()

state_space = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)
state_space_goal = RealVectorPastTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)


planner_name = Planners.RRT_CONNECT
planners = compile_all_planners(world, state_space_start, state_space_goal)
planner = planners[planner_name]
problem = PlanningProblem(planner)


start_config = world.robot.config
state_start = np.append(start_config, 0)
goal_config = world.robot.goal_state

state_goal = np.append(goal_config, TIME_HORIZON)
# state_goal = np.append(start_config, TIME_HORIZON)

goal_region.set_goal_state(state_goal)
# planner.local_planner.plan(state_src=np.array([0.2236, 32]), state_dst=np.array([0.3272, 37]), space=state_space_start)

path_sparse, data = problem.solve(
    state_start,
    goal_region,
    min_planning_time=0,
    max_planning_time=np.inf,
    max_iters=100,
)
print("Planning done, pre processing path")
path = planner.local_planner.interpolate_path_from_waypoints(path_sparse)
print(path)

# print(path)
# print(path[1:, 0] - path[:-1, 0])
# path_pp = post_process_path_space_time_minimal_time(
#     state_space_start,
#     planner.local_planner,
#     goal_region,
#     path,
#     max_cnt_no_improvement=10
# )

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
world.create_space_time_map(time_horizon=TIME_HORIZON)
world.render_configuration_space(ax, time_horizon=TIME_HORIZON)


if planner_name in (Planners.RRT_STAR_CONNECT_PARTIAL, Planners.RRT_CONNECT, Planners.RRT_STAR_INFORMED_CONNECT_PARTIAL):
    tree_data = ((problem.planner.tree_start, "blue"), (problem.planner.tree_goal, "red"))
else:
    tree_data = ((problem.planner.tree, "blue"), )


def draw_unit_ball(planner):
    CL, x_center = planner.CL, planner.x_center
    thetas = np.linspace(0, np.pi * 2, 100)
    xs = np.vstack([np.cos(thetas), np.sin(thetas)])
    x_center = x_center.reshape(-1, 1)
    xs = CL @ xs + x_center
    return xs.T


for tree, color in tree_data:
    verts, edges = tree.get_vertices(), tree.get_edges()
    render_tree(ax, verts, edges, color=color)


if path.size > 0:
    ax.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")
    ax.plot(path_sparse[:, 0], path_sparse[:, 1], color="black", label="path", lw=2, ls="--", marker=".")

# if path_pp.size > 0:
#     ax.plot(path_pp[:, 0], path_pp[:, 1], color="red", label="path", lw=2, ls="--", marker=".")

goal_region_r = goal_region.radius
goal_region_xy_lower_corner = (goal_region.state[0] - goal_region_r, 0)

ax.add_patch(
    Rectangle(
        goal_region_xy_lower_corner,
        width=goal_region_r * 2,
        height=TIME_HORIZON,
        alpha=0.1,
        color="red"
    )
)

polygon_values = np.array([
    tuple(state_start),
    (state_start[0] + world.robot.max_actuation * TIME_HORIZON, TIME_HORIZON),
    (state_start[0] - world.robot.max_actuation * TIME_HORIZON, TIME_HORIZON),
])

ax.add_patch(
    Polygon(polygon_values, alpha=0.1)
)


if path.size and planner_name == Planners.RRT_STAR_INFORMED_CONNECT_PARTIAL:
    vals = draw_unit_ball(planner)
    ax.plot(vals[:, 0], vals[:, 1])

plt.show()
