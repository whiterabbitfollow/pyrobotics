from examples.moving.moving_world import MovingBox1DimWorld
from pyrb.mp.planners.moving.rrt_connect import RRTConnectPlannerTimeVarying
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(14)  # Challenging, solvable with ~200 steps...

PLANNING_TIME = 1
TIME_HORIZON = 60

world = MovingBox1DimWorld()
world.reset()

planner = RRTConnectPlannerTimeVarying(
    world,
    local_planner_nr_coll_steps=10,
    local_planner_max_distance=0.5,
    max_nr_vertices=int(1e5)
)

start_state = np.append(world.robot.config, 0)
goal_state = np.append(world.robot.goal_state, TIME_HORIZON)

path, status = planner.plan(
    start_state,
    goal_state,
    max_planning_time=PLANNING_TIME
)

world.create_space_time_map(time_horizon=TIME_HORIZON)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

world.render_configuration_space(ax, time_horizon=TIME_HORIZON)

for tree, color in ((planner.tree_start, "blue"), (planner.tree_goal, "red")):
    vert_cnt = tree.vert_cnt
    verts = tree.vertices
    ax.scatter(verts[:vert_cnt, 0], verts[:vert_cnt, 1], color="black")
    for i_parent, indxs_children in tree.edges_parent_to_children.items():
        for i_child in indxs_children:
            q = np.stack([verts[i_parent], verts[i_child]], axis=0)
            ax.plot(q[:, 0], q[:, 1], color=color)

if path.size > 0:
    ax.plot(path[:, 0], path[:, 1], color="orange", label="path", lw=2, ls="--", marker=".")

plt.show()
