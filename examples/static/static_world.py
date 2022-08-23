import pyrb

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from dataclasses import dataclass

from pyrb.mp.base_world import BaseMPWorld
from pyrb.rendering.utils import robot_configuration_to_matplotlib_rectangles
from pyrb.robot import Manipulator


class StaticBoxObstacle:

    def __init__(self):
        self.width = np.random.uniform(0.1, 0.3)
        self.height = np.random.uniform(0.1, 0.3)
        self.angle = np.random.uniform(0, np.pi)
        self.transform = None
        self.geometry = None

    def reset(self):
        theta_box = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1, 1.0)
        self.pose = np.array([radius * np.cos(theta_box), radius * np.sin(theta_box)])
        R = pyrb.kin.rot_z_to_SO3(self.angle)
        p = np.append(self.pose, 0)
        self.transform = pyrb.kin.rot_trans_to_SE3(R, p)
        self.geometry = trimesh.creation.box(extents=(self.width, self.height, 0.1), transform=self.transform)


@dataclass
class Range:
    lower: float
    upper: float

    def to_tuple(self):
        return self.lower, self.upper


@dataclass
class WorldData:
    x: Range
    y: Range

    def __init__(self, xs, ys):
        self.x = Range(*xs)
        self.y = Range(*ys)


class RandomBoxesObstacleRegion:

    def __init__(self, world_data, obstacle_free_region=None):
        self.world_data: WorldData = world_data
        nr_static_obstacles = 10
        self.obstacles = [StaticBoxObstacle() for _ in range(nr_static_obstacles)]
        self.obstacle_free_region = obstacle_free_region
        self.collision_manager = trimesh.collision.CollisionManager()

    def reset(self, obstacle_free_region=None):
        for i, obs in enumerate(self.obstacles):
            while True:
                obs.reset()
                is_free = obstacle_free_region is None
                is_free = is_free or not obstacle_free_region.in_collision_single(obs.geometry)
                if is_free:
                    self.collision_manager.add_object(f"box_{i}", obs.geometry)
                    break


class StaticBoxesWorld(BaseMPWorld):

    def __init__(self):
        joint_limits = [
            [-2 * np.pi / 3, 2 * np.pi / 3],
            [-np.pi + np.pi / 4, 0]
        ]
        robot_data = {
            "links": [
                {
                    "geometry": {
                        "width": 0.3,
                        "height": 0.1,
                        "depth": 0.1,
                        "direction": 0
                    }
                },
                {
                    "geometry": {
                        "width": 0.3,
                        "height": 0.1,
                        "depth": 0.1,
                        "direction": 0
                    }
                }
            ],
            "joints": [
                {
                    "position": np.array([0.0, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": joint_limits[0]
                },
                {
                    "position": np.array([0.3, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": joint_limits[1]
                }
            ],
            "end_effector": {
                "position": np.array([0.3, 0.0, 0.0])
            }
        }
        data = WorldData((-1, 1), (-1, 1))
        robot = Manipulator(robot_data)
        obstacles = RandomBoxesObstacleRegion(data)
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def reset(self):
        self.robot.set_config(np.array([np.pi / 2 + np.pi / 10, -np.pi / 2 - np.pi / 10]))
        self.obstacles.reset(obstacle_free_region=self.robot.collision_manager)

    def view(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.render_world(ax1)
        self.render_configuration_space(ax2)
        plt.show()

    def render_world(self, ax):
        rectangles = []
        rectangles_params = robot_configuration_to_matplotlib_rectangles(self.robot)
        for rectangles_param in rectangles_params:
            rectangles.append(Rectangle(*rectangles_param))
        ax.add_collection(PatchCollection(rectangles, color="blue", edgecolor="black"))

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

    def render_configuration_space(self, ax):
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
        ax.set_title("Configuration space, $\mathcal{C}$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        joint_limits = self.robot.joint_limits
        ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
        ax.set_ylim(joint_limits[1, 0], joint_limits[1, 1])



