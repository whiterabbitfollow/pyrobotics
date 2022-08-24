from matplotlib.patches import Rectangle

from examples.moving.moving_world import MovingBox1DimWorld
from examples.utils import render_tree
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt import RRTPlanner, LocalPlannerSpaceTime
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace
from pyrb.mp.utils.trees.tree_rewire import TreeRewireSpaceTime

np.random.seed(14)  # Challenging, solvable with ~200 steps...
PLANNING_TIME = 10
TIME_HORIZON = 60

world = MovingBox1DimWorld()
world.reset()

state_space = RealVectorTimeSpace(world, world.robot.nr_joints, world.robot.joint_limits, time_horizon=TIME_HORIZON)
goal_region = RealVectorTimeGoalRegion()

local_planner = LocalPlannerSpaceTime(
    state_space.world.robot.max_actuation,
    nr_time_steps=5,
    min_path_distance=.3,
    min_coll_step_size=0.05,
    max_distance=(1.0, 10)
)

planner = RRTPlanner(
    space=state_space,
    tree=TreeRewireSpaceTime(
        local_planner=local_planner,
        space=state_space,
        nearest_radius=1.0,
        nearest_time_window=10,
        max_nr_vertices=int(1e4)
    ),
    local_planner=local_planner
)

problem = PlanningProblem(
    planner=planner
)

state_start = np.append(world.robot.config, 0)
goal_config = world.robot.goal_state
state_goal = np.append(goal_config, TIME_HORIZON)
goal_region.set_goal_state(state_goal)

path, status = problem.solve(
    state_start,
    goal_region,
    max_iters=200
)

print(status.time_taken, status.status)

world.create_space_time_map(time_horizon=TIME_HORIZON)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

world.render_configuration_space(ax, time_horizon=TIME_HORIZON)

tree, color = planner.tree, "blue"
verts, edges = tree.get_vertices(), tree.get_edges()

render_tree(ax, verts, edges)

if path.size > 0:
    ax.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")

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

plt.show()

