import numpy as np
import tqdm

from examples.space_time.agent_and_adv.agent_n_adversary_world import AgentAdversary2DWorld
from examples.space_time.make import compile_all_planners, Planners
from pyrb.mp.problem import PlanningProblem
from pyrb.mp.utils.goal_regions import RealVectorMinimizingTimeGoalRegion
from pyrb.mp.utils.spaces import RealVectorTimeSpace, RealVectorPastTimeSpace




from collections import defaultdict


np.random.seed(6)
MIN_PLANNING_TIME = 5
MAX_PLANNING_TIME = 10
TIME_HORIZON = 300

world = AgentAdversary2DWorld()

goal_region = RealVectorMinimizingTimeGoalRegion()

state_space_start = RealVectorTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)
state_space_goal = RealVectorPastTimeSpace(
    world, world.robot.nr_joints, world.robot.joint_limits, max_time=TIME_HORIZON, goal_region=goal_region
)
# goal_region = RealVectorTimeGoalRegion()


planners = compile_all_planners(world, state_space_start, state_space_goal)

planners = {name.value: planners[name] for name in (Planners.RRT, Planners.RRT_STAR, Planners.RRT_CONNECT, Planners.RRT_STAR_CONNECT_PARTIAL)}

stats = {planner_name: defaultdict(list) for planner_name in planners}

problem = PlanningProblem(None)

for i in tqdm.tqdm(range(50)):
    world.reset()
    state_start = np.append(world.robot.config, 0)
    goal_config = world.robot.goal_state
    state_goal = np.append(goal_config, TIME_HORIZON)
    for planner_name, planner in planners.items():
        problem.set_planner(planner)
        goal_region.set_goal_state(state_goal)
        path, data = problem.solve(
            state_start,
            goal_region,
            min_planning_time=MIN_PLANNING_TIME,
            max_planning_time=MAX_PLANNING_TIME
        )
        stats[planner_name]["success"].append(data.status == "success")
        stats[planner_name]["time_elapsed"].append(data.meta_data_problem["time_elapsed"])
        stats[planner_name]["time_first_found"].append(data.meta_data_problem["time_first_found"])
        stats[planner_name]["path_cost"].append(data.meta_data_planner["cost_best_path"])
        stats[planner_name]["path_len"].append(path.shape[0] if path.shape[0] else np.inf)
        stats[planner_name]["path_time_horizon"].append(path[-1, -1] if path.size else np.inf)


for planner_name in stats:
    print(planner_name)
    for stat_name in ("success", "time_elapsed", "time_first_found", "path_cost", "path_len", "path_time_horizon"):
        vals = np.array(stats[planner_name][stat_name])
        vals_finite = vals[np.isfinite(vals)]
        print(stat_name, np.mean(vals_finite))



# # TODO: Could be the case that ingest to many states
# print(status.time_taken, status.status)
# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# vertices = planner.tree.get_vertices()
#
#
# if path.size:
#     for i, state in enumerate(path):
#         config = state[:-1]
#         t = state[-1]
#         world.robot.set_config(config)
#         world.set_time(t)
#         print(world.get_current_config_smallest_obstacle_distance())
#         world.render_world(ax1)
#         sub_path = path[:i + 1, :-1]
#         world.render_configuration_space(ax2, path=sub_path)
#         curr_verts = vertices[:, -1] == t
#         if curr_verts.any():
#             ax2.scatter(vertices[curr_verts, 0], vertices[curr_verts, 1])
#         fig.suptitle(f"Time {t}")
#         plt.pause(0.1)
#         ax1.cla()
#         ax2.cla()
# else:
#     for t in range(TIME_HORIZON):
#         world.robot.set_config(np.array([np.pi / 2, 0]))
#         world.set_time(t)
#         world.render_world(ax1)
#         world.render_configuration_space(ax2)
#         curr_verts = (vertices[:, -1] - t) == 0
#         if curr_verts.any():
#             ax2.scatter(vertices[curr_verts, 0], vertices[curr_verts, 1])
#         plt.pause(0.01)
#         ax1.cla()
#         ax2.cla()
#         fig.suptitle(f"Time {t} distance: {world.get_current_config_smallest_obstacle_distance():.2f}")
#


# fig, ax1 = plt.subplots(1, 1)
# world.render_world(ax1)
# plt.show()


# start_state = np.append(world.robot.config, 0)
# goal_config = world.robot.goal_state

# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# t = 0
# world.robot.set_config(start_state[:-1])
# world.set_time(t)
# world.render_world(ax1)
# world.render_configuration_space(ax2)
# fig.suptitle(f"Time {t} distance: {world.get_current_config_smallest_obstacle_distance():.2f}")
# plt.show()
#
#
# path, status = planner.plan(
#     start_state,
#     goal_config,
#     max_planning_time=180,
#     min_planning_time=20,
#     time_horizon=300
# )

# path_1, _ = RRTStarPlannerTimeVarying(
#     world,
#     local_planner_nr_coll_steps=2,
#     local_planner_max_distance=np.inf,
#     max_nr_vertices=int(1e5)
# ).plan(
#     start_state,
#     goal_config,
#     max_planning_time=30
# )
#
#
# planner = RRTStarPlannerTimeVaryingModified(
#     world,
#     local_planner_nr_coll_steps=2,
#     local_planner_max_distance=np.inf,
#     max_nr_vertices=int(1e5)
# )
# path_2, _ = planner.plan(
#     start_state,
#     goal_config,
#     max_planning_time=50,
#     time_horizon=60
# )


# planner = ModifiedRRTPlannerTimeVarying(
#     world,
#     local_planner_nr_coll_steps=2,
#     local_planner_max_distance=np.inf,
#     max_nr_vertices=int(1e5)
# )
#
# path_3, _ = planner.plan(
#     start_state,
#     goal_config,
#     max_planning_time=60,
#     min_planning_time=59
# )
#
# def path_len(p):
#     return np.linalg.norm(p[1:, :] - p[:-1, :], axis=1).sum()
#
# # print(path_2)
# path = path_2
# print(path_len(path_2), path_len(path_3))# 35.29198395465184

# TODO: RRTConnect!
# state_start, config_goal, max_planning_time = np.inf
# print(path, goal_config, world.robot.goal_state)
# print(goal_config)
# import matplotlib.pyplot as plt
# import trimesh
# import pyrb
# mesh = trimesh.creation.cylinder(0.1, height=planner.time_horizon)
# offset = np.append(goal_config, planner.time_horizon/2)
# mesh.apply_transform(pyrb.kin.rot_trans_to_SE3(p=offset))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# i = planner.vert_cnt
#
# # ax.scatter(planner.vertices[:i, 0], planner.vertices[:i, 1], planner.vertices[:i, 2])
#
# for i_parent, indxs_children in planner.edges_parent_to_children.items():
#     for i_child in indxs_children:
#         q = np.stack([planner.vertices[i_parent], planner.vertices[i_child]], axis=0)
#         ax.plot(q[:, 0], q[:, 1], q[:, 2], ls="-", marker=".", color="black")
# ax.plot_trisurf(
#     mesh.vertices[:, 0],
#     mesh.vertices[:, 1],
#     triangles=mesh.faces,
#     Z=mesh.vertices[:, 2],
#     color="red",
#     alpha=0.1
# )
# plt.show()
# print(status.status, status.time_taken, status.nr_verts)
# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# if path.size:
#     for i, state in enumerate(path):
#         config = state[:-1]
#         t = state[-1]
#         world.robot.set_config(config)
#         world.set_time(t)
#         print(world.get_current_config_smallest_obstacle_distance())
#         world.render_world(ax1)
#         sub_path = path[:i + 1, :-1]
#         world.render_configuration_space(ax2, path=sub_path)
#         curr_verts = planner.vertices[:, -1] == t
#         if curr_verts.any():
#             ax2.scatter(planner.vertices[curr_verts, 0], planner.vertices[curr_verts, 1])
#         fig.suptitle(f"Time {t}")
#         plt.pause(0.1)
#         ax1.cla()
#         ax2.cla()
# else:
#     for t in range(600):
#         world.robot.set_config(np.array([np.pi / 2, 0]))
#         world.set_time(t)
#         world.render_world(ax1)
#         world.render_configuration_space(ax2)
#         curr_verts = (planner.vertices[:, -1] - t) == 0
#         if curr_verts.any():
#             ax2.scatter(planner.vertices[curr_verts, 0], planner.vertices[curr_verts, 1])
#         plt.pause(0.01)
#         ax1.cla()
#         ax2.cla()
#         fig.suptitle(f"Time {t} distance: {world.get_current_config_smallest_obstacle_distance():.2f}")
