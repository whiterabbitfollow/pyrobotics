from worlds.robot_cylinder.world import RobotCylinderWorld
from worlds.robot_s_and_d.render import render_trajectory_in_world
import matplotlib.pyplot as plt
import numpy as np



def interpolate_path(conf_s, conf_g):
    total_distance = np.linalg.norm(conf_g - conf_s)
    max_actuation = 0.1
    delta_distances = np.arange(0, total_distance, max_actuation)
    if delta_distances[-1] != total_distance:
        delta_distances = np.append(delta_distances, total_distance)
    betas = delta_distances/total_distance
    nr_steps = betas.size
    confs_s = np.tile(conf_s.reshape(1, -1), (nr_steps, 1))
    confs_g = np.tile(conf_g.reshape(1, -1), (nr_steps, 1))
    confs = (confs_s.T * (1-betas) + confs_g.T * betas).T
    return confs


def categorize_collision(world):
    obstacles = world.obstacles
    robot = world.robot
    time_steps = trajectory[:, 0].astype(int)
    configs = trajectory[:, 1:]
    dynamic_collision = False
    static_collision = False
    collision = False
    for t, config in zip(time_steps, configs):
        world.set_time(t)
        robot.set_config(config)
        collision, names = robot.collision_manager.in_collision_other(obstacles.collision_manager, return_names=True)
        names = [o for r, o in names]
        if collision:
            dynamic_collision = any("dynamic" in n for n in names)
            static_collision += any("static" in n for n in names)
            break
    return collision, static_collision, dynamic_collision


world = RobotCylinderWorld()
import pickle


with open("empty_world.pckl", "wb") as fp:
    pickle.dump(world, fp)



# world.reset()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# cnt = 0
# collision_cnt = 0
# static_collision_cnt = 0
# dynamic_collision_cnt = 0
#
#
# while True:
#     t_start = np.random.randint(0, 100)
#     world.set_time(t_start)
#     config_s = world.sample_collision_free_state()
#     config_g = world.sample_collision_free_state()
#     configs = interpolate_path(config_s, config_g)
#     ts = np.arange(0, configs.shape[0]).reshape(-1, 1) + t_start
#     trajectory = np.hstack([ts, configs])
#     render_trajectory_in_world(ax, world, trajectory)
#     collision, static_collision, dynamic_collision = categorize_collision(world)
#     cnt += 1
#     collision_cnt += collision
#     static_collision_cnt += static_collision
#     dynamic_collision_cnt += dynamic_collision
#     print(
#         " ".join(
#             ["%s: %.2f" % (name, c / cnt) for c, name in
#              zip(
#                  [collision_cnt, static_collision_cnt, dynamic_collision_cnt],
#                  ["coll", "static_coll", "dyn_coll"]
#              )
#              ]
#         )
#     )
