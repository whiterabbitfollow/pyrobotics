from matplotlib.patches import Rectangle

from examples.moving.moving_world import MovingBox1DimWorld
from examples.utils import render_tree, plot_rrt_connect_planner_results
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.mp.planners.rrt import LocalPlannerSpaceTime
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace
from pyrb.mp.utils.trees.tree import Tree
from pyrb.mp.utils.trees.tree_rewire import TreeRewireSpaceTime

np.random.seed(14)  # Challenging, solvable with ~200 steps...
PLANNING_TIME = 10
TIME_HORIZON = 60

world = MovingBox1DimWorld()
world.reset()

state_space_start = RealVectorTimeSpace(world, world.robot.nr_joints, world.robot.joint_limits, time_horizon=TIME_HORIZON)
local_planner = LocalPlannerSpaceTime(
    min_path_distance=0.2,
    min_coll_step_size=0.05,
    max_distance=(1.0, 5),
    max_actuation=world.robot.max_actuation
)

tree_start = Tree(
    space=state_space_start,
    max_nr_vertices=int(1e4),
    vertex_dim=state_space_start.dim
)


state_space_goal = RealVectorPastTimeSpace(world, world.robot.nr_joints, world.robot.joint_limits, time_horizon=TIME_HORIZON)

tree_goal = Tree(
    space=state_space_goal,
    max_nr_vertices=int(1e4),
    vertex_dim=state_space_goal.dim
)


planner = RRTConnectPlanner(
    local_planner=local_planner,
    tree_start=tree_start,
    tree_goal=tree_goal
)

problem = PlanningProblem(
    planner=planner
)


goal_region = RealVectorTimeGoalRegion()
state_start = np.append(world.robot.config, 0)
goal_config = world.robot.goal_state
state_goal = np.append(goal_config, TIME_HORIZON)
goal_region.set_goal_state(state_goal)


path, status = problem.solve(
    state_start,
    goal_region,
    min_planning_time=2,
    max_planning_time=np.inf
)

# # TODO: Could be the case that ingest to many states
print(status.time_taken, status.status)

world.create_space_time_map(time_horizon=TIME_HORIZON)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

world.render_configuration_space(ax, time_horizon=TIME_HORIZON)

for tree, color, lw in ((planner.tree_start, "blue", 4), (planner.tree_goal, "red", 2)):
    vertices, edges = tree.get_vertices(), tree.get_edges()
    render_tree(ax, vertices, edges, color=color, lw=lw)

# if path.size > 0:
#     ax.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")

goal_region_r = planner.goal_region.radius
goal_region_xy_lower_corner = (goal_config[0] - goal_region_r, 0)

ax.add_patch(
    Rectangle(
        goal_region_xy_lower_corner,
        width=goal_region_r*2,
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


# render_tree(ax2, tree_start.get_vertices(), tree_start.get_edges())
# render_tree(ax3, tree_goal.get_vertices(), tree_goal.get_edges())

plt.show()
