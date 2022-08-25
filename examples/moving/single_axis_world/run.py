from matplotlib.patches import Rectangle

from examples.moving.make import make_rrt, make_rrt_connect, compile_all_planners
from examples.moving.moving_world import MovingBox1DimWorld
from examples.utils import render_tree
from pyrb.mp.planners.problem import PlanningProblem
from pyrb.mp.planners.rrt import RRTPlanner, LocalPlannerSpaceTime
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion, RealVectorMinimizingTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace
from pyrb.mp.utils.trees.tree import Tree

np.random.seed(14)  # Challenging, solvable with ~200 steps...



world = MovingBox1DimWorld()
world.reset()


PLANNING_TIME = 10
TIME_HORIZON = 60
state_space = RealVectorTimeSpace(world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON)
# goal_region = RealVectorTimeGoalRegion()
goal_region = RealVectorMinimizingTimeGoalRegion()

state_space_start = RealVectorTimeSpace(world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON)
state_space_goal = RealVectorPastTimeSpace(world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON)

planner_name = "rrt_star"
planners = compile_all_planners(world, state_space_start, state_space_goal)
planner = planners[planner_name]
problem = PlanningProblem(planner)

state_start = np.append(world.robot.config, 0)
goal_config = world.robot.goal_state
state_goal = np.append(goal_config, TIME_HORIZON)
goal_region.set_goal_state(state_goal)

path, status = problem.solve(
    state_start,
    goal_region,
    min_planning_time=1,
    max_planning_time=PLANNING_TIME
)


# TODO: Could be the case that ingest to many states
print(status.time_taken, status.status)
world.create_space_time_map(time_horizon=TIME_HORIZON)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

world.render_configuration_space(ax, time_horizon=TIME_HORIZON)

tree = problem.planner.tree_start if "connect" in planner_name else problem.planner.tree
color = "blue"
verts, edges = tree.get_vertices(), tree.get_edges()

render_tree(ax, verts, edges)

if path.size > 0:
    ax.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")

goal_region_r = goal_region.radius
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

