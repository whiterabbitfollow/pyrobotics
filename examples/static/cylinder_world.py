import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from dataclasses import dataclass

from pyrb.mp.base_world import BaseMPWorld


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


class CylinderObstacle:

    def __init__(self):
        self.radius = 0.3
        self.height = 0.1
        self.geometry = trimesh.creation.cylinder(radius=self.radius, height=self.height)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("cylinder", self.geometry)
        self.pos = np.array([0, 0])


class PointMassRobot:

    def __init__(self):
        self.state_dim = 2
        self.radius = 0.1
        self.height = 0.1
        self.geometry = trimesh.creation.cylinder(radius=self.radius, height=self.height)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("cylinder", self.geometry)
        self.config = np.array([0, 0])
        self.transform = np.eye(4)
        self.joint_limits = np.array([[-1, 1], [-1, 1]])

    def set_config(self, config):
        self.config = config
        self.transform[:2, -1] = config
        self.collision_manager.set_transform(name="cylinder", transform=self.transform)


class CylinderWorld(BaseMPWorld):

    def __init__(self):
        data = WorldData((-1, 1), (-1, 1))
        robot = PointMassRobot()
        obstacles = CylinderObstacle()
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def reset(self):
        self.robot.set_config(np.array([-.5, 0]))

    def view(self):
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        self.render_world(ax1)
        # self.render_configuration_space(ax2)
        plt.show()

    def render_world(self, ax):
        ax.add_patch(Circle(xy=self.obstacles.pos, radius=self.obstacles.radius, color="red"))
        ax.add_patch(Circle(xy=self.robot.config, radius=self.robot.radius, color="blue"))
        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.y.to_tuple())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("World, $\mathcal{W} = \mathbb{R}^2$")
        ax.set_aspect("equal")


if __name__ == "__main__":
    world = CylinderWorld()
    world.reset()
    print(world.get_min_distance_to_obstacle())
    # world.is_collision_free_transition()
    world.view()
