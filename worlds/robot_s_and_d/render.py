import numpy as np
import matplotlib.pyplot as plt
import trimesh

import pyrb
from pyrb.rendering.utils import render_robot_configuration_meshes


def render_trajectory_in_world(ax, world, trajectory):
    obstacles = world.obstacles
    robot = world.robot
    time_steps = trajectory[:, 0].astype(int)
    configs = trajectory[:, 1:]
    config_start = configs[0, :]
    config_end = configs[-1, :]

    for t, config in zip(time_steps, configs):
        world.set_time(t)
        robot.set_config(config_start)
        render_robot_configuration_meshes(ax, robot, color="orange", alpha=0.5)
        robot.set_config(config_end)
        render_robot_configuration_meshes(ax, robot, color="blue", alpha=0.5)
        robot.set_config(config)
        collision, names = robot.collision_manager.in_collision_other(obstacles.collision_manager, return_names=True)
        if collision:
            color = "red"
        else:
            color = "blue"
        render_robot_configuration_meshes(ax, robot, color=color, alpha=1.0)
        obstacles_in_collision = [obstacle_name for _, obstacle_name in names]
        render_obstacle_manager_meshes(ax, obstacles, t, obstacles_in_collision)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.pause(0.1)
        ax.cla()


def render_obstacle_manager_meshes(ax, om, ts, obstacles_in_collision=None):
    obstacles_in_collision = obstacles_in_collision or []
    for o in om.static_obstacles:
        m = o.mesh
        if o.name in obstacles_in_collision:
            color = "red"
            alpha = 1.0
        else:
            color = "green"
            alpha = 0.5
        render_mesh(ax, m, color=color, alpha=alpha)

    for o in om.dynamic_obstacles:
        m = o.mesh
        T = o.get_transform_at_time_step(ts)
        m.apply_transform(T)
        # ax.plot(o.positions[:, 0], o.positions[:, 1], o.positions[:, 2])
        m = o.mesh
        if o.name in obstacles_in_collision:
            color = "red"
            alpha = 1.0
        else:
            color = "green"
            alpha = 0.5
        render_mesh(ax, m, color=color, alpha=alpha)
        R = T[:3, :3]
        p = T[:3, 3]
        for e, c in zip(R.T, ["red", "green", "blue"]):
            line = np.vstack([p, p + e * 0.5])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], c=c)
        m.apply_transform(pyrb.kin.SE3_inv(T))


def render_mesh(ax, mesh, **kwargs):
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        triangles=mesh.faces,
        Z=mesh.vertices[:, 2],
        **kwargs
    )


def render_obstacle_space(ax, world):
    limits = world.robot.joint_limits
    thetas = np.linspace(limits[:, 0], limits[:, 1], 20).T
    theta1, theta2, theta3 = thetas
    w, h, d = np.diff(thetas).T[0]
    T1, T2, T3 = np.meshgrid(theta1, theta2, theta3)
    configs = np.hstack([T1.reshape(-1, 1), T2.reshape(-1, 1), T3.reshape(-1, 1)])
    meshes = []
    for config in configs:
        world.robot.set_config(config)
        collision, objs_in_collision = world.robot.collision_manager.in_collision_other(
            world.obstacles.collision_manager, return_names=True
        )
        if collision:
            dynamic_collision = any("dynamic" in n for _, n in objs_in_collision)
            static_collision = any("static" in n for _, n in objs_in_collision)
            # config_colls.append(config)
            box = trimesh.creation.box(extents=(w, h, d))
            # objs_in_collision
            T = pyrb.kin.rot_trans_to_SE3(p=config)
            box.apply_transform(T)
            meshes.append((static_collision, dynamic_collision, box))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    for static_collision, dynamic_collision, box in meshes:
        if static_collision and dynamic_collision:
            color = "red"
        elif static_collision:
            color = "green"
        elif dynamic_collision:
            color = "blue"
        else:
            color = "black"
        render_mesh(ax, box, color=color)
    ax.set_xlim(limits[0, 0], limits[0, 1])
    ax.set_ylim(limits[1, 0], limits[1, 1])
    ax.set_zlim(limits[2, 0], limits[2, 1])
    # plt.show()
