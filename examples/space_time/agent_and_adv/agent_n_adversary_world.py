import copy
from matplotlib.patches import Circle
import numpy as np

from examples.space_time.actors.adversary import Mobile2DOFAdversaryManipulator
from examples.data.manipulators import DATA_MANIPULATOR_2DOF
from pyrb.mp.base_agent import MotionPlanningAgentActuated
from pyrb.mp.base_world import BaseMPTimeVaryingWorld, WorldData2D
from pyrb.rendering.utils import robot_configuration_to_patch_collection


class AgentAdversary2DWorld(BaseMPTimeVaryingWorld):

    def __init__(self, robot=None, obstacles=None):
        data = WorldData2D((-1, 1), (-1, 1))
        robot_data = copy.deepcopy(DATA_MANIPULATOR_2DOF)
        robot = robot or MotionPlanningAgentActuated(robot_data, max_actuation=0.1)
        obstacles = obstacles or Mobile2DOFAdversaryManipulator()
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def render_world(self, ax):
        curr_config = self.robot.config.copy()
        goal_config = self.robot.goal_state
        self.robot.set_config(goal_config)
        ax.add_collection(robot_configuration_to_patch_collection(self.robot, color="blue", alpha=0.1))
        self.robot.set_config(curr_config)
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
        self.reset_config(seed=seed)

    def reset_config(self, seed=None):
        random_state = None
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        goal_state = self.sample_collision_free_state()
        self.robot.set_goal_state(goal_state)
        self.start_config = self.sample_collision_free_state()
        self.robot.set_config(self.start_config)
        if seed is not None:
            np.random.set_state(random_state)

    def render_configuration_space(self, ax, path=None):
        joint_limits = self.robot.joint_limits
        theta_raw_1 = np.linspace(joint_limits[0, 0], joint_limits[0, 1], 100)
        theta_raw_2 = np.linspace(joint_limits[1, 0], joint_limits[1, 1], 100)
        theta_grid_1, theta_grid_2 = np.meshgrid(theta_raw_1, theta_raw_2)
        thetas = np.stack([theta_grid_1.ravel(), theta_grid_2.ravel()], axis=1)
        collision_mask = []
        for theta in thetas:
            self.robot.set_config(theta)
            collision = self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)
            collision_mask.append(not collision)
        collision_mask = np.array(collision_mask).reshape(100, 100)
        ax.pcolormesh(theta_grid_1, theta_grid_2, collision_mask)
        ax.scatter(self.start_config[0], self.start_config[1], label="Config", s=100)
        ax.scatter(self.robot.goal_state[0], self.robot.goal_state[1], s=100)
        ax.add_patch(Circle(tuple(self.robot.goal_state), radius=0.1, color="red", alpha=0.1, label="Goal set"))
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
