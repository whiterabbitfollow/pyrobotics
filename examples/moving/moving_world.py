import copy

import trimesh
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import numpy as np

import pyrb
from examples.data.manipulators import DATA_MANIPULATOR_2DOF, DATA_MANIPULATOR_1DOF
from examples.utils import render_manipulator_on_axis
from pyrb.mp.base_agent import MotionPlanningAgentActuated
from pyrb.mp.base_world import WorldData2D, BaseMPTimeVaryingWorld


class StaticBoxObstacle:

    def __init__(self, pos_xy,  width, height, angle):
        self.width = width
        self.height = height
        self.angle = angle
        self.position = np.append(pos_xy, 0)
        R = pyrb.kin.rot_z_to_SO3(self.angle)
        self.transform = pyrb.kin.rot_trans_to_SE3(R, self.position)
        self.geometry = trimesh.creation.box(extents=(self.width, self.height, 0.1), transform=self.transform)


class MovingBoxObstacle:

    def __init__(self, pos_xy,  width, height, angle, amplitude=0.5, angular_freq=(np.pi * 2) / 20):
        self.width = width
        self.height = height
        self.angle = angle
        self.amplitude = amplitude
        self.angular_freq = angular_freq
        self.position_start = np.append(pos_xy, 0)
        self.R = pyrb.kin.rot_z_to_SO3(self.angle)
        self.transform = pyrb.kin.rot_trans_to_SE3(self.R, self.position_start)
        self.geometry = trimesh.creation.box(extents=(self.width, self.height, 0.1))
        self.time_step = 0

    def set_time(self, t):
        self.time_step = t
        position = self.position_start.copy()
        # position[0] += np.random.random() * 0.01
        # position[1] = self.position_start[1] + np.cos(self.angular_freq * t) * self.amplitude + np.random.random() * 0.5
        position[1] = self.position_start[1] + np.cos(self.angular_freq * t) * self.amplitude #  + np.random.random() * 0.5
        self.transform = pyrb.kin.rot_trans_to_SE3(self.R, position)


class BoxesObstacles:

    def __init__(self, world_data, static_obstacles=None, moving_obstacles=None, obstacle_free_region=None):
        self.world_data: WorldData2D = world_data
        self.static_obstacle = static_obstacles or []
        self.moving_obstacles = moving_obstacles or []
        self.obstacle_free_region = obstacle_free_region
        self.collision_manager = trimesh.collision.CollisionManager()
        self.obstacles = self.static_obstacle + self.moving_obstacles
        for i, obs in enumerate(self.static_obstacle):
            self.collision_manager.add_object(name=f"static_box_{i}", mesh=obs.geometry)
        for i, obs in enumerate(self.moving_obstacles):
            self.collision_manager.add_object(name=f"moving_box_{i}", mesh=obs.geometry)

    def set_time(self, t):
        for i, obs in enumerate(self.moving_obstacles):
            obs.set_time(t)
            self.collision_manager.set_transform(name=f"moving_box_{i}", transform=obs.transform)


class MovingBox1DimWorld(BaseMPTimeVaryingWorld):

    def __init__(self):
        data = WorldData2D((-1, 1), (-1, 1))
        robot_data = copy.deepcopy(DATA_MANIPULATOR_1DOF)
        robot = MotionPlanningAgentActuated(robot_data, max_actuation=0.1)
        obstacles = BoxesObstacles(
            world_data=data,
            moving_obstacles=[MovingBoxObstacle(
                pos_xy=[0.351, -0.3],
                width=0.1,
                height=0.25,
                angle=0,
                amplitude=0.4,
                angular_freq=(np.pi * 2) / 40
            )
            ]
        )
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def reset(self):
        self.start_config = np.array([-np.pi/3])
        self.robot.set_goal_state(np.array([np.pi/3]))
        self.robot.set_config(self.start_config)

    def view(self):
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        render_world(self, ax1)
        # self.render_configuration_space(ax2)
        plt.show()

    def create_space_time_map(self, t_start=0, time_horizon=100):
        joint_limits = self.robot.joint_limits
        theta_1_nr_data_points = 100
        thetas_raw_1 = np.linspace(joint_limits[0, 0], joint_limits[0, 1], theta_1_nr_data_points)
        time = np.arange(t_start + 1, time_horizon + 1)
        theta_grid_1, time_grid_2 = np.meshgrid(thetas_raw_1, time)
        states = np.stack([theta_grid_1.ravel(), time_grid_2.ravel()], axis=1)
        collision_mask = []
        for state in states:
            self.robot.set_config(state[:-1])
            self.obstacles.set_time(state[-1])
            collision = self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)
            collision_mask.append(not collision)
        collision_mask = np.array(collision_mask).reshape(time_horizon-t_start, theta_1_nr_data_points)
        self.cmap = theta_grid_1, time_grid_2, collision_mask

    def render_configuration_space(self, ax, t_start=0, time_horizon=100):
        joint_limits = self.robot.joint_limits
        ax.pcolormesh(*self.cmap)
        # ax.scatter(self.start_config[0], self.start_config[1], label="Config")
        # ax.add_patch(Circle(tuple(self.robot.goal_state), radius=0.1, color="red", alpha=0.1))
        # ax.scatter(self.robot.goal_state[0], self.robot.goal_state[1], label="Goal config")
        # if path is not None and path.size > 0:
        #     ax.plot(path[:, 0], path[:, 1], ls="-", marker=".", label="path")
        ax.set_title("Configuration space, $\mathcal{C}$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"time")
        ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
        ax.set_ylim(t_start, time_horizon + 5)
        # ax.legend(loc="best")


class MovingBoxWorld(BaseMPTimeVaryingWorld):

    def __init__(self):
        data = WorldData2D((-1, 1), (-1, 1))
        robot_data = copy.deepcopy(DATA_MANIPULATOR_2DOF)
        robot = MotionPlanningAgentActuated(robot_data, max_actuation=0.1)
        obstacles = BoxesObstacles(
            world_data=data,
            static_obstacles=[StaticBoxObstacle([-0.1, 0.45], 0.1, 0.3, 2 * np.pi/3)],
            moving_obstacles=[MovingBoxObstacle([0.4, 0.3], 0.1, 0.5, 0)]
        )
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def reset(self):
        self.start_config = np.array([np.pi / 2 - np.pi / 10, -np.pi / 10])
        self.robot.set_goal_state(np.array([-np.pi / 2 + np.pi / 10, -np.pi / 10]))
        self.robot.set_config(self.start_config)

    def view(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.render_world(ax1)
        self.render_configuration_space(ax2)
        plt.show()

    def render_world(self, ax):
        render_world(self, ax)

    def make_configuration_space_map(self):
        limits = self.robot.joint_limits
        thetas_raw_1 = np.linspace(limits[0, 0], limits[0, 1], 100)
        thetas_raw_2 = np.linspace(limits[1, 0], limits[1, 1], 100)
        theta_grid_1, theta_grid_2, *_ = np.meshgrid(thetas_raw_1, thetas_raw_2)
        thetas = np.stack([theta_grid_1.ravel(), theta_grid_2.ravel()], axis=1)
        collision_mask = []
        for theta in thetas:
            self.robot.set_config(theta)
            collision = self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)
            collision_mask.append(not collision)
        collision_mask = np.array(collision_mask).reshape(100, 100)
        return theta_grid_1, theta_grid_2, collision_mask

    def render_configuration_space(self, ax, path=None):
        joint_limits = self.robot.joint_limits
        theta_grid_1, theta_grid_2, collision_mask = self.make_configuration_space_map()
        ax.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)
        ax.scatter(self.start_config[0], self.start_config[1], label="Config")
        ax.add_patch(Circle(tuple(self.robot.goal_state), radius=0.1, color="red", alpha=0.1))
        ax.scatter(self.robot.goal_state[0], self.robot.goal_state[1], label="Goal config")
        if path is not None and path.size > 0:
            ax.plot(path[:, 0], path[:, 1], ls="-", marker=".", label="path")
        ax.set_title("Configuration space, $\mathcal{C}$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
        ax.set_ylim(joint_limits[1, 0], joint_limits[1, 1])
        ax.legend(loc="best")


def render_world(world, ax):
    curr_config = world.robot.config.copy()
    goal_config = world.robot.goal_state
    world.robot.set_config(goal_config)
    color = "blue"
    render_manipulator_on_axis(ax, world.robot, color=color, alpha=0.1)

    world.robot.set_config(curr_config)
    color = "blue"
    render_manipulator_on_axis(ax, world.robot, color=color)

    for angle_rad, link in zip(world.robot.config, world.robot.links):
        xy = link.frame[:2, 3]
        ax.scatter(xy[0], xy[1], color="black")

    static_obstacles = []
    for obs in world.obstacles.obstacles:
        p_local = np.array([-obs.width / 2, -obs.height / 2, 0])
        p_global = pyrb.kin.SE3_mul(obs.transform, p_local)
        static_obstacles.append(Rectangle(tuple(p_global)[:2], obs.width, obs.height, angle=np.rad2deg(obs.angle)))

    ax.add_collection(PatchCollection(static_obstacles, color="green"))
    ax.set_xlim(*world.data.x.to_tuple())
    ax.set_ylim(*world.data.y.to_tuple())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("World, $\mathcal{W} = \mathbb{R}^2$")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    world = MovingBoxWorld()
    world.reset()

    # render_world(world, ax1)
    # # world.render_configuration_space(ax2)
    # world.set_time(t)
    # plt.pause(0.1)
    # ax1.cla()
    # # ax2.cla()