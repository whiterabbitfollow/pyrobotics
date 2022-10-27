from matplotlib.patches import Rectangle

from examples.space_time.make import compile_all_planners, Planners
from examples.space_time.moving_world import MovingBox1DimWorld
from examples.utils import render_tree
import numpy as np
import matplotlib.pyplot as plt

from pyrb.mp.problem import PlanningProblem
from pyrb.mp.utils.goal_regions import RealVectorTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace
from pyrb.mp.utils.trees.tree import Tree



def render(ax, world, goal_region, sub_paths, future_paths, t_start, t_end):

    start_config = world.robot.config
    goal_config = world.robot.goal_state

    state_start = np.append(start_config, t_start)
    state_goal = np.append(goal_config, t_end)
    goal_region.set_goal_state(state_goal)

    world.create_space_time_map(t_start=t_start, time_horizon=t_end)

    world.render_configuration_space(ax, t_start=t_start, time_horizon=t_end)
    # tree = problem.planner.tree_start
    tree = problem.planner.tree_start

    verts, edges = tree.get_vertices(), tree.get_edges()
    render_tree(ax, verts, edges)
    path = np.vstack(sub_paths)

    if path.size > 0:
        ax.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")

    if future_paths.size:
        ax.plot(future_paths[:, 0], future_paths[:, 1], color="red", label="path", lw=2, ls="--", marker=".")

    goal_region_r = goal_region.radius
    goal_region_xy_lower_corner = (goal_region.state[0] - goal_region_r, t_start)

    ax.add_patch(
        Rectangle(
            goal_region_xy_lower_corner,
            width=goal_region_r * 2,
            height=TIME_HORIZON,
            alpha=0.1,
            color="red"
        )
    )

    #
    # polygon_values = np.array([
    #     tuple(state_start),
    #     (state_start[0] + world.robot.max_actuation * TIME_HORIZON, TIME_HORIZON),
    #     (state_start[0] - world.robot.max_actuation * TIME_HORIZON, TIME_HORIZON),
    # ])
    #
    # ax.add_patch(
    #     Polygon(polygon_values, alpha=0.1)
    # )



frames = []



np.random.seed(14)  # Challenging, solvable with ~200 steps...

world = MovingBox1DimWorld()
world.reset()

PLANNING_TIME = 10
TIME_HORIZON = 60
mpc_nr_time_steps = 2


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
max_actuation = world.robot.max_actuation

sub_paths = []

t_end = TIME_HORIZON

fig, ax = plt.subplots(1, 1, figsize=(10, 10))


cnt = 0

import tqdm

max_cnt = 50
tbar = tqdm.tqdm(total=max_cnt)
while cnt < max_cnt:


    t_start = t
    t_end = TIME_HORIZON + t
    state_start = np.append(start_config, t_start)
    state_goal = np.append(goal_config, t_end)
    goal_region.set_goal_state(state_goal)

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

    sub_path = path[:mpc_nr_time_steps]
    sub_paths.append(sub_path)

    vertex_root = sub_path[-1]

    # target_start_config, t_target = vertex_root[:-1], vertex_root[-1].astype(int)
    # start_config = np.clip(target_start_config - start_config, -max_actuation, max_actuation) + start_config
    # t = t + 1
    start_config, t = vertex_root[:-1], vertex_root[-1].astype(int)

    tree = planner.tree_start

    i_goal = planner.get_goal_state_index()
    indxs = tree.find_path_indices_to_root_from_vertex_index(i_goal)
    indxs.reverse()
    indxs = indxs[mpc_nr_time_steps-1:]
    indx_map = {indxs[0]: 0}

    vertices, edges = tree.get_vertices(), tree.get_edges()

    render_tree(ax, vertices, edges)
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

    # planner.tree_goal.clear()
    render(ax, world, goal_region, sub_paths, path[mpc_nr_time_steps-1:], t_start, t_end)
    # plt.pause(1.0)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
    ax.cla()
    cnt += 1
    tbar.update()
tbar.close()

import moviepy.editor as mpy
clip = mpy.ImageSequenceClip(frames, fps=3)  # 2 seconds
clip.write_videofile("fixed_time_horizon_problem_continuous.mp4", fps=3)

