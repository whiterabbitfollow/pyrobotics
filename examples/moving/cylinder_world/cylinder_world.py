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
    t: Range

    def __init__(self, xs, ts):
        self.x = Range(*xs)
        self.t = Range(*ts)


class CylinderObstacle:

    def __init__(self):
        self.radius = 2
        self.pos = np.array([0, 5])

    def set_time(self, t):
        self.t = t


class PointMassRobot:

    def __init__(self):
        self.state_dim = 1
        self.radius = 0.1
        self.config = np.array([0])
        self.joint_limits = np.array([[-10, 10]])
        self.max_actuation = 1

    def set_config(self, config):
        self.config = config


class CylinderWorld(BaseMPWorld):

    def __init__(self):
        data = WorldData((-10, 10), (0, 10))
        robot = PointMassRobot()
        obstacles = CylinderObstacle()
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def reset(self):
        self.robot.set_config(np.array([0, 0]))

    def view(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self.render_configuration_space(ax)
        plt.show()

    def render_configuration_space(self, ax):
        ax.add_patch(Circle(xy=self.obstacles.pos, radius=self.obstacles.radius, color="red"))
        ax.add_patch(Circle(xy=self.robot.config, radius=self.robot.radius, color="blue"))
        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.t.to_tuple())
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("t")
        ax.set_title("World, $\mathcal{W} = \mathbb{R}^2$")
        ax.set_aspect("equal")

    def is_collision_free_state(self, state) -> bool:
        self.robot.set_config(state)
        return np.linalg.norm(self.obstacles.pos - state) > self.obstacles.radius

    def get_min_distance_to_obstacle(self) -> float:
        return max(np.linalg.norm(self.obstacles.pos - self.robot.config) - self.obstacles.radius, 0)


if __name__ == "__main__":
    world = CylinderWorld()
    world.reset()
    world.robot.set_config(np.array([0.3, 3]))
    print(world.get_min_distance_to_obstacle())
    world.view()
