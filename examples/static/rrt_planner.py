import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from examples.static.static_world import StaticBoxesWorld
from examples.utils import plot_rrt_planner_results, render_rrt_planner_results, render_tree
from pyrb.mp.problem import PlanningProblem
from pyrb.mp.planners.rrt import RRTPlanner
from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
from pyrb.mp.utils.spaces import RealVectorStateSpace
from pyrb.mp.utils.trees.tree import Tree

from pyrb.mp.planners.local_planners import LocalPlanner
from pyrb.mp.utils.trees.tree_rewire import TreeRewire

np.random.seed(16)

world = StaticBoxesWorld()
world.reset()

state_space = RealVectorStateSpace(
    world,
    world.robot.nr_joints,
    world.robot.joint_limits
)

goal_region = RealVectorGoalRegion()

planner = RRTPlanner(
    space=state_space,
    tree=Tree(max_nr_vertices=int(1e3), vertex_dim=state_space.dim),
    local_planner=LocalPlanner(
        min_coll_step_size=0.01,
        max_distance=0.5
    )
)

problem = PlanningProblem(
    planner=planner
)

for i in range(14):
    state_start = state_space.sample_collision_free_state()
    state_goal = state_space.sample_collision_free_state()

goal_region.set_goal_state(state_goal)

path, status = problem.solve(
    state_start, goal_region, min_planning_time=0, max_planning_time=1
)


def render_results(ax, title):
    world.render_configuration_space(ax)
    vertices, edges = planner.tree.get_vertices(), planner.tree.get_edges()
    render_tree(ax, vertices, edges)
    ax.scatter(state_start[0], state_start[1], color="green", label="start, $q_I$")
    ax.scatter(state_goal[0], state_goal[1], color="red", label="goal, $q_G$")
    if path.size > 0:
        ax.plot(path[:, 0], path[:, 1], color="orange", label="path", ls="-", marker=".")
    ax.add_patch(Circle(state_goal, radius=goal_region.radius, alpha=0.2, color="red"))
    ax.add_patch(Circle(state_start, radius=0.04, color="green"))
    ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_aspect("equal")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))


render_results(ax1, title="RRT")





local_planner = LocalPlanner(
    min_coll_step_size=0.1,
    max_distance=0.5
)

planner = RRTPlanner(
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

path, status = problem.solve(
    state_start, goal_region, min_planning_time=1, max_planning_time=10
)

render_results(ax2, title="RRT*")


fig.tight_layout()
plt.show()


# plot_rrt_planner_results(
#     world, planner, path, state_start, state_goal, goal_region
# )
