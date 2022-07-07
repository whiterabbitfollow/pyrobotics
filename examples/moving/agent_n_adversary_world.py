import numpy as np

from examples.moving.actors.adversary import Mobile2DOFAdversaryManipulator
from examples.moving.actors.agents import Manipulator2DOF
from examples.utils import render_manipulator_on_axis
from pyrb.mp.base_world import BaseMPTimeVaryingWorld, WorldData2D


class AgentAdversary2DWorld(BaseMPTimeVaryingWorld):

    def __init__(self):
        data = WorldData2D((-1, 1), (-1, 1))
        robot = Manipulator2DOF()
        obstacles = Mobile2DOFAdversaryManipulator()
        super().__init__(robot=robot, data=data, obstacles=obstacles)

    def render_world(self, ax):
        curr_config = self.robot.config.copy()
        goal_config = self.robot.goal_state
        self.robot.set_config(goal_config)

        color = "blue"
        render_manipulator_on_axis(ax, self.robot, color=color)
        self.robot.set_config(curr_config)
        color = "blue"
        render_manipulator_on_axis(ax, self.robot, color=color, alpha=0.1)

        color = "green"
        render_manipulator_on_axis(ax, self.obstacles, color=color)

        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.y.to_tuple())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    def reset(self):
        self.obstacles.reset()
        goal_state = self.sample_collision_free_state()
        self.robot.set_goal_state(goal_state)

        start_state = self.sample_collision_free_state()
        self.robot.set_config(start_state)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    world = AgentAdversary2DWorld()
    world.reset()
    world.render_world(ax1)
    plt.show()

