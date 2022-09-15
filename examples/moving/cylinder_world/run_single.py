from matplotlib.patches import Rectangle

from examples.moving.cylinder_world.cylinder_world import CylinderWorld
from examples.moving.make import compile_all_planners, Planners
from examples.moving.moving_world import MovingBox1DimWorld
from examples.utils import render_tree
from pyrb.mp.planners.local_planners import LocalPlannerSpaceTime
from pyrb.mp.planners.post_processing import post_process_path_space_time_minimal_time
from pyrb.mp.planners.problem import PlanningProblem
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from pyrb.mp.planners.rrt import RRTPlanner
from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion, RealVectorMinimizingTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace
from pyrb.mp.utils.trees.tree import Tree
from pyrb.mp.utils.trees.tree_rewire import TreeRewireSpaceTime

np.random.seed(14)  # Challenging, solvable with ~200 steps...

world = CylinderWorld()
world.reset()


PLANNING_TIME = 10
TIME_HORIZON = 10

#
# 1.04719755

# goal_region = RealVectorMinimizingTimeGoalRegion()
goal_region = RealVectorTimeGoalRegion(radius=1)

state_space = RealVectorTimeSpace(
    world, world.robot.state_dim, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

state_space_start = RealVectorTimeSpace(
    world, world.robot.state_dim, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)
state_space_goal = RealVectorPastTimeSpace(
    world, world.robot.state_dim, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

planner_name = Planners.RRT_STAR

local_planner = LocalPlannerSpaceTime(
        world.robot.max_actuation,
        # nr_time_steps=5,
        # min_path_distance=.3,
        min_coll_step_size=0.05,
        max_distance=(1.0, 3),
        include_distance_to_obst=True
    )

planner = RRTPlanner(
    space=state_space,
    tree=TreeRewireSpaceTime(
        local_planner=local_planner,
        space=state_space,
        nearest_radius=4.0,
        nearest_time_window=5,
        max_nr_vertices=int(1e5)
    ),
    local_planner=local_planner
)


problem = PlanningProblem(planner)
start_config = np.array([0])
state_start = np.append(start_config, 0)
goal_config = np.array([0])

state_goal = np.append(goal_config, TIME_HORIZON)
# state_goal = np.append(start_config, TIME_HORIZON)

goal_region.set_goal_state(state_goal)
# planner.local_planner.plan(state_src=np.array([0.2236, 32]), state_dst=np.array([0.3272, 37]), space=state_space_start)


path, data = problem.solve(
    state_start,
    goal_region,
    min_planning_time=0,
    max_planning_time=5
)


#
# planner.tree.set_root_vertex(np.array(state_start))
#
#
# s1 = np.array([0, 0])
# s2 = np.array([-2, 2])
# status, path, transition_data = local_planner.plan(state_space_start, s1, s2)
# print(transition_data)
# planner.tree.append_vertex(s2, 0, 0)


print("Planning done, pre processing path")

path_pp = np.array([])


# path_pp = post_process_path_space_time_minimal_time(
#     state_space_start,
#     planner.local_planner,
#     goal_region,
#     path,
#     max_cnt_no_improvement=10
# )

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

world.render_configuration_space(ax)


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


if path_pp.size > 0:
    ax.plot(path_pp[:, 0], path_pp[:, 1], color="red", label="path", lw=2, ls="--", marker=".")

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

plt.show()

