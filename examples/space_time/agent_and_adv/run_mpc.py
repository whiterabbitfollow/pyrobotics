from examples.space_time.agent_and_adv.agent_n_adversary_world import AgentAdversary2DWorld
from examples.space_time.make import compile_all_planners, Planners
import numpy as np
import matplotlib.pyplot as plt

from pyrb.mp.problem import PlanningProblem
from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace
from pyrb.mp.utils.trees.tree import Tree
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# if path.size:
#     for i, state in enumerate(path):




def plot(ax1, ax2, world, goal_region, sub_paths, t_end, future_path):

    start_config = world.robot.config
    goal_config = world.robot.goal_state
    state_goal = np.append(goal_config, t_end)
    goal_region.set_goal_state(state_goal)

    path = np.vstack(sub_paths)

    config = path[-1, :-1]
    t = path[-1, -1]

    world.robot.set_config(config)
    world.set_time(t)
    world.render_world(ax1)
    world.render_configuration_space(ax2)


    ax2.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")
    ax2.plot(future_path[:, 0], future_path[:, 1], color="red", label="path", lw=2, ls="--", marker=".")
    fig.suptitle(f"Time {t}")




np.random.seed(14)  # Challenging, solvable with ~200 steps...

world = AgentAdversary2DWorld()
world.reset()

PLANNING_TIME = 10
TIME_HORIZON = 300


mpc_nr_time_steps = 4


goal_region = RealVectorTimeGoalRegion()
state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)

state_space_goal = RealVectorPastTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON
)


planner_name = Planners.RRT_STAR_INFORMED_CONNECT_PARTIAL
planners = compile_all_planners(world, state_space_start, state_space_goal)
planner = planners[planner_name]
problem = PlanningProblem(planner)

t = 0
start_config = world.robot.config
goal_config = world.robot.goal_state

sub_paths = []

t_end = TIME_HORIZON

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

while True:

    t_start = t
    t_end = TIME_HORIZON + t
    state_start = np.append(start_config, t_start)


    state_goal = np.append(goal_config, t_end)
    goal_region.set_goal_state(state_goal)

    print(t_start, goal_region.is_config_within(state_start))

    state_space_start.min_time = t_start
    state_space_start.max_time = t_end

    state_space_goal.min_time = t_start
    state_space_goal.max_time = t_end

    path, status = problem.solve(
        state_start,
        goal_region,
        min_planning_time=0.5,
        max_planning_time=5,
        clear=False
    )

    future_path = path[mpc_nr_time_steps:]
    sub_path = path[:mpc_nr_time_steps]
    sub_paths.append(sub_path)

    vertex_root = sub_path[-1]
    start_config, t = vertex_root[:-1], vertex_root[-1].astype(int)
    tree = planner.tree_start

    i_goal = planner.get_goal_state_index()
    indxs = tree.find_path_indices_to_root_from_vertex_index(i_goal)
    indxs.reverse()
    indxs = indxs[mpc_nr_time_steps-1:]
    indx_map = {indxs[0]: 0}

    vertices, edges = tree.get_vertices(), tree.get_edges()
    tree_new = Tree(
        space=state_space_start,
        max_nr_vertices=int(1e4),
        vertex_dim=state_space_start.dim
    )
    tree_new.add_vertex(vertex_root)

    for i_parent, i_child in zip(indxs[:-1], indxs[1:]):
        edge_cost = state_space_start.transition_cost(vertices[i_child], vertices[i_parent])
        i_parent_new = indx_map[i_parent]
        i_new = tree_new.append_vertex(vertices[i_child], i_parent=i_parent_new, edge_cost=edge_cost)
        indx_map[i_child] = i_new

    planner.tree_start = tree_new
    planner.tree_goal.clear()
    planner.found_path = False
    planner.connected = False

    plot(ax1, ax2, world, goal_region, sub_paths, t_start, future_path)
    plt.pause(1.0)
    ax1.cla()
    ax2.cla()
