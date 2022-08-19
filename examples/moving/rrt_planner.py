from examples.moving.moving_world import MovingBoxWorld
from pyrb.mp.planners.moving.rrt import RRTPlannerTimeVarying

import numpy as np
import matplotlib.pyplot as plt

world = MovingBoxWorld()
world.reset()
time_horizon = 60

planner = RRTPlannerTimeVarying(
    world,
    local_planner_max_distance=1.0,
    local_planner_nr_coll_steps=2
)


state_start = np.append(world.start_config, 0)
config_goal = world.robot.goal_state
path, status = planner.plan(
    state_start=state_start,
    config_goal=config_goal,
    time_horizon=time_horizon
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
world.reset()
print(status.status, status.time_taken)

iterable = enumerate(path) if path.size > 0 else enumerate([world.start_config] * time_horizon)

for i, state in iterable:
    config = state[:-1]
    t = state[-1]
    world.robot.set_config(config)
    world.set_time(t)
    world.render_world(ax1)
    sub_path = path[:i, :-1]
    world.render_configuration_space(ax2, path=sub_path)
    curr_verts = (planner.vertices[:, -1] - t) == 0
    if curr_verts.any():
        ax2.scatter(planner.vertices[curr_verts, 0], planner.vertices[curr_verts, 1])
    plt.pause(0.1)
    ax1.cla()
    ax2.cla()
