import pyrb

import trimesh
import numpy as np

from dataclasses import dataclass

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


class StaticBoxesWorld:

    def __init__(self):
        robot_data = {
            "links": [
                {
                    "geometry": {
                        "height": 0.15,
                        "width": 0.5
                    }
                },
                {
                    "geometry": {
                        "height": 0.15,
                        "width": 0.4
                    }
                }
            ],
            "joints": [
                {
                    "position": np.array([0.0, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": (-np.pi/2, np.pi/2)
                },
                {
                    "position": np.array([0.5, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": (-np.pi / 2, np.pi / 2)
                }
            ],
            "end_effector": {
                "position": np.array([0.2, 0.0, 0.0])
            }
        }
        self.data = WorldData((-1, 1), (-1, 1))
        self.robot = Manipulator(robot_data)
        self.obstacle_region = RandomBoxesObstacleRegion(self.data)

    def reset(self):
        self.robot.set_config(np.array([np.pi / 2 + np.pi / 10, -np.pi / 2 - np.pi / 10]))
        self.obstacle_region.reset(obstacle_free_region=self.robot.collision_manager)

    def is_collision_free(self, q):
        self.robot.set_config(q)
        return not self.robot.collision_manager.in_collision_other(self.obstacle_region.collision_manager)
