import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from examples.space_time.actors.adversary import Mobile2DOFAdversaryManipulator
from examples.data.manipulators import DATA_MANIPULATOR_2DOF
from pyrb.mp.base_agent import MotionPlanningAgentActuated
from pyrb.mp.base_world import BaseMPWorld, WorldData2D
from pyrb.rendering.utils import robot_configuration_to_patch_collection


class ReplanningAgentAdversary2DWorld(BaseMPWorld):

    def __init__(self, robot=None, obstacles=None):
        data = WorldData2D((-1, 1), (-1, 1))
        robot_data = copy.deepcopy(DATA_MANIPULATOR_2DOF)
        robot = robot or MotionPlanningAgentActuated(robot_data, max_actuation=0.1)
        obstacles = obstacles or Mobile2DOFAdversaryManipulator()
        self.start_config = None
        self.t = 0
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def render_world(self, ax):
        self.robot.set_config(self.goal_config)
        ax.add_collection(robot_configuration_to_patch_collection(self.robot, color="blue", alpha=0.1))
        self.robot.set_config(self.start_config)
        ax.add_collection(robot_configuration_to_patch_collection(self.robot, color="blue"))
        ax.add_collection(robot_configuration_to_patch_collection(self.obstacles, color="green"))
        # ax.set_xlim(*self.data.x.to_tuple())
        # ax.set_ylim(*self.data.y.to_tuple())
        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.y.to_tuple())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(r"World, $\mathcal{W} = \mathbb{R}^2$")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    def reset(self, seed=None):
        self.obstacles.reset(seed)
        random_state = None
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        self.goal_config = self.sample_collision_free_state()
        self.start_config = self.sample_collision_free_state()
        self.robot.set_config(self.start_config)
        self.t = 0  # TODO: sample time?
        self.obstacles.set_time(self.t)
        if seed is not None:
            np.random.set_state(random_state)

    def set_start_config(self, start_config):
        self.start_config = start_config

    def step(self, delta_config):
        # TODO: could check delta_config???
        config_nxt = self.start_config + delta_config
        collision_free = self.is_collision_free_time_step_transition(
            state_src=self.start_config,
            state_dst=config_nxt
        )
        if collision_free:
            self.start_config = config_nxt
        return collision_free

    def is_collision_free_time_step_transition(self, state_src, state_dst, nr_coll_steps=10):
        is_collision_free = False
        t_src = self.t
        t_dst = self.t + 1
        for i in range(1, nr_coll_steps + 1):
            beta = i / nr_coll_steps
            state = (1 - beta) * state_src + beta * state_dst
            t = (1 - beta) * t_src + beta * t_dst
            self.obstacles.set_time(t)
            is_collision_free = self.is_collision_free_state(state)
            if not is_collision_free:
                break
        return is_collision_free

    def set_time(self, t):
        self.t = t
        self.obstacles.set_time(t)

    def render_configuration_space(self, ax, path=None, resolution=100):
        joint_limits = self.robot.joint_limits
        theta_raw_1 = np.linspace(joint_limits[0, 0], joint_limits[0, 1], resolution)
        theta_raw_2 = np.linspace(joint_limits[1, 0], joint_limits[1, 1], resolution)
        theta_grid_1, theta_grid_2 = np.meshgrid(theta_raw_1, theta_raw_2)
        thetas = np.stack([theta_grid_1.ravel(), theta_grid_2.ravel()], axis=1)
        collision_mask = []
        for theta in thetas:
            self.robot.set_config(theta)
            collision = self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)
            collision_mask.append(not collision)
        collision_mask = np.array(collision_mask).reshape(resolution, resolution)
        ax.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)
        if self.start_config is not None:
            ax.scatter(self.start_config[0], self.start_config[1], label="config")
        if self.goal_config is not None:
            ax.scatter(self.goal_config[0], self.goal_config[1], s=100)
            ax.add_patch(Circle(tuple(self.goal_config), radius=0.1, color="red", alpha=0.1, label="Goal set"))
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], ls="-", marker=".", label="path")
        ax.set_title(r"Configuration space $\mathcal{C} \in \mathbb{R}^2$")
        ax.set_xlabel(r"$\theta_1$ [rad]")
        ax.set_ylabel(r"$\theta_2$ [rad]")
        ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
        ax.set_ylim(joint_limits[1, 0], joint_limits[1, 1])
        ax.set_aspect("equal")
        ax.legend(loc="best")

    def get_current_config_smallest_obstacle_distance(self):
        distance = self.robot.collision_manager.min_distance_other(self.obstacles.collision_manager)
        return distance

    def view(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.render_world(ax1)
        self.render_configuration_space(ax2)
        plt.show()

