from examples.moving.moving_world import MovingBoxWorld
from pyrb.mp.planners.moving.rrt import RRTPlannerTimeVarying

import numpy as np
import matplotlib.pyplot as plt

world = MovingBoxWorld()
world.reset()
planner = RRTPlannerTimeVarying(world, time_horizon=30, max_actuation=0.1)
state_start = np.append(world.start_config, 0)
config_goal = world.goal_config
path, status = planner.plan(state_start=state_start, config_goal=config_goal)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
world.reset()

assert path.size > 0, "No solution found"

for i, state in enumerate(path):
    config = state[:-1]
    t = state[-1]
    print(world.is_collision_free_state(state))
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

state = path[-1]
config = state[:-1]
t = state[-1]

world.robot.set_config(config)
world.render_world(ax1)
world.render_configuration_space(ax2, path=path)
world.set_time(t)
plt.show()

