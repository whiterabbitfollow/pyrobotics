
import trimesh
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import numpy as np

import pyrb
from pyrb.mp.base_world import BaseMPWorld, WorldData2D, BaseMPTimeVaryingWorld
from pyrb.rendering.utils import robot_configuration_to_matplotlib_rectangles
from examples.moving.actors.agents import Manipulator2DOF


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

    def __init__(self, pos_xy,  width, height, angle):
        self.width = width
        self.height = height
        self.angle = angle
        self.position_start = np.append(pos_xy, 0)
        self.R = pyrb.kin.rot_z_to_SO3(self.angle)
        self.transform = pyrb.kin.rot_trans_to_SE3(self.R, self.position_start)
        self.geometry = trimesh.creation.box(extents=(self.width, self.height, 0.1))
        self.time_step = 0

    def set_time(self, t):
        w = 20 / (np.pi * 2)
        self.time_step = t
        position = self.position_start.copy()
        position[1] *= np.cos(t/w)
        self.transform = pyrb.kin.rot_trans_to_SE3(self.R, position)


class BoxesObstacles:

    def __init__(self, world_data, obstacle_free_region=None):
        self.world_data: WorldData2D = world_data
        self.static_obstacle = [
            StaticBoxObstacle([-0.1, 0.45], 0.1, 0.3, 2 * np.pi/3)
        ]
        self.moving_obstacles = [
            MovingBoxObstacle([0.4, 0.3], 0.1, 0.5, 0)
        ]
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



class MovingBoxWorld(BaseMPTimeVaryingWorld):

    def __init__(self):
        data = WorldData2D((-1, 1), (-1, 1))
        robot = Manipulator2DOF()
        obstacles = BoxesObstacles(data)
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def reset(self):
        self.start_config = np.array([np.pi / 2 - np.pi / 10, -np.pi / 10])
        self.goal_config = np.array([-np.pi / 2 + np.pi / 10, -np.pi / 10])
        self.robot.set_config(self.start_config)

    def view(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.render_world(ax1)
        self.render_configuration_space(ax2)
        plt.show()

    def render_world(self, ax):
        self.robot.set_config(config)
        rectangles_params = robot_configuration_to_matplotlib_rectangles(self.robot)
        for rectangles_param in rectangles_params:
            rectangles.append(Rectangle(*rectangles_param))
        coll = PatchCollection(rectangles)
        coll.set_color("blue")
        coll.set_edgecolor("black")
        ax.add_collection(coll)

        for angle_rad, link in zip(self.robot.config, self.robot.links):
            xy = link.frame[:2, 3]
            ax.scatter(xy[0], xy[1], color="black")

        static_obstacles = []
        for obs in self.obstacles.obstacles:
            p_local = np.array([-obs.width / 2, -obs.height / 2, 0])
            p_global = pyrb.kin.SE3_mul(obs.transform, p_local)
            static_obstacles.append(Rectangle(tuple(p_global)[:2], obs.width, obs.height, angle=np.rad2deg(obs.angle)))
        ax.add_collection(PatchCollection(static_obstacles, color="green"))

        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.y.to_tuple())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("World, $\mathcal{W} = \mathbb{R}^2$")

    def render_configuration_space(self, ax, path=None):
        thetas_raw = np.linspace(-np.pi, np.pi, 100)
        theta_grid_1, theta_grid_2 = np.meshgrid(thetas_raw, thetas_raw)
        thetas = np.stack([theta_grid_1.ravel(), theta_grid_2.ravel()], axis=1)
        collision_mask = []
        for theta in thetas:
            self.robot.set_config(theta)
            collision = self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)
            collision_mask.append(not collision)

        collision_mask = np.array(collision_mask).reshape(100, 100)
        ax.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)
        ax.scatter(self.start_config[0], self.start_config[1], label="Config")
        ax.add_patch(Circle(tuple(self.goal_config), radius=0.1, color="red", alpha=0.1))
        ax.scatter(self.goal_config[0], self.goal_config[1], label="Goal config")
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], ls="-", marker=".", label="path")
        ax.set_title("Configuration space, $\mathcal{C}$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        joint_limits = self.robot.joint_limits
        ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
        ax.set_ylim(joint_limits[1, 0], joint_limits[1, 1])
        ax.legend(loc="best")


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    world = MovingBoxWorld()
    world.reset()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    for t in range(100):
        world.reset()
        world.render_world(ax1)
        world.render_configuration_space(ax2)
        world.set_time(t)
        plt.pause(0.1)
        ax1.cla()
        ax2.cla()
