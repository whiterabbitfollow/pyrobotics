from examples.moving.moving_world import MovingBox1DimWorld
from pyrb.mp.planners.moving.time_optimal.rrt import RRTPlannerTimeVarying
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(14)  # Challenging, solvable with ~200 steps...

PLANNING_TIME = 1
TIME_HORIZON = 60

world = MovingBox1DimWorld()
world.reset()

# planner = RRTPlannerTimeVarying(
#     world,
#     time_horizon=300,
#     local_planner_nr_coll_steps=10,
#     local_planner_max_distance=np.inf,
#     max_nr_vertices=int(1e5)
# )

# planner = ModifiedRRTPlannerTimeVarying(
#     world,
#     time_horizon=TIME_HORIZON,
#     local_planner_nr_coll_steps=10,
#     local_planner_max_distance=np.inf,
#     max_nr_vertices=int(1e5)
# )

planner = RRTPlannerTimeVarying(
    world,
    local_planner_nr_coll_steps=10,
    local_planner_max_distance=np.inf,
    max_nr_vertices=int(1e5)
)

start_config = np.append(world.robot.config, 0)
goal_config = world.robot.goal_state

path, status = planner.plan(
    start_config,
    goal_config,
    max_planning_time=PLANNING_TIME,
    time_horizon=TIME_HORIZON
)
print(path, status.status)

world.create_space_time_map(time_horizon=TIME_HORIZON)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

i = planner.vert_cnt
world.render_configuration_space(ax, time_horizon=TIME_HORIZON)

ax.scatter(planner.vertices[:i, 0], planner.vertices[:i, 1])
for i_parent, indxs_children in planner.edges_parent_to_children.items():
    for i_child in indxs_children:
        q = np.stack([planner.vertices[i_parent], planner.vertices[i_child]], axis=0)
        ax.plot(q[:, 0], q[:, 1], ls="-", marker=".", color="black")


goal_region_r = planner.goal_region_radius
goal_region_xy_lower_corner = (goal_config[0] - goal_region_r, 0)

ax.add_patch(Rectangle(goal_region_xy_lower_corner, width=goal_region_r*2, height=TIME_HORIZON, alpha=0.1, color="red"))
plt.show()
