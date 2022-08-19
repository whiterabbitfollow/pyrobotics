import copy

import numpy as np
from matplotlib.patches import Circle

from examples.moving.actors.adversary import Mobile2DOFAdversaryManipulator
from examples.utils import render_manipulator_on_axis
from examples.data.manipulators import DATA_MANIPULATOR_2DOF
from pyrb.mp.base_agent import MotionPlanningAgentActuated
from pyrb.mp.base_world import BaseMPTimeVaryingWorld, WorldData2D
# from pyrb.mp.planners.moving.rrt import ModifiedRRTPlannerTimeVarying
# from pyrb.mp.planners.moving.rrt_star import RRTStarPlannerTimeVarying
# from pyrb.mp.planners.moving.rrt_star_dev import RRTStarPlannerTimeVaryingModified
from scratches.rrt_moving.rrt_star_dev_1 import RRTStarPlannerTimeVaryingModified


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
        render_manipulator_on_axis(ax, self.robot, color="blue", alpha=0.1)

        self.robot.set_config(curr_config)
        render_manipulator_on_axis(ax, self.robot, color="blue")

        render_manipulator_on_axis(ax, self.obstacles, color="green")
        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.y.to_tuple())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect("equal")

    def reset(self):
        self.obstacles.reset()
        self.reset_config()

    def reset_config(self):
        goal_state = self.sample_collision_free_state()
        self.robot.set_goal_state(goal_state)
        self.start_config = self.sample_collision_free_state()
        self.robot.set_config(self.start_config)

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
        ax.add_patch(Circle(tuple(self.robot.goal_state), radius=0.1, color="red", alpha=0.1))
        ax.scatter(self.robot.goal_state[0], self.robot.goal_state[1], label="Goal config")
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], ls="-", marker=".", label="path")
        ax.set_title("Configuration space, $\mathcal{C}$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        joint_limits = self.robot.joint_limits
        ax.set_xlim(joint_limits[0, 0], joint_limits[0, 1])
        ax.set_ylim(joint_limits[1, 0], joint_limits[1, 1])
        ax.set_aspect("equal")
        ax.legend(loc="best")

    def get_current_config_smallest_obstacle_distance(self):
        distance = self.robot.collision_manager.min_distance_other(self.obstacles.collision_manager)
        return distance


if __name__ == "__main__":
    # from pyrb.mp.planners.moving.rrt_star import RRTStarPlannerTimeVarying
    # 0
    np.random.seed(6)  # Challenging, solvable with ~200 steps...
    world = AgentAdversary2DWorld()
    world.reset()
    # world.render_world(ax1)
    # plt.show()

    start_state = np.append(world.robot.config, 0)
    goal_config = world.robot.goal_state

    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # t = 0
    # world.robot.set_config(start_state[:-1])
    # world.set_time(t)
    # world.render_world(ax1)
    # world.render_configuration_space(ax2)
    # fig.suptitle(f"Time {t} distance: {world.get_current_config_smallest_obstacle_distance():.2f}")
    # plt.show()
    #
    #
    # path, status = planner.plan(
    #     start_state,
    #     goal_config,
    #     max_planning_time=180,
    #     min_planning_time=20,
    #     time_horizon=300
    # )

    # path_1, _ = RRTStarPlannerTimeVarying(
    #     world,
    #     local_planner_nr_coll_steps=2,
    #     local_planner_max_distance=np.inf,
    #     max_nr_vertices=int(1e5)
    # ).plan(
    #     start_state,
    #     goal_config,
    #     max_planning_time=30
    # )
    #

    planner = RRTStarPlannerTimeVaryingModified(
        world,
        local_planner_nr_coll_steps=2,
        local_planner_max_distance=np.inf,
        max_nr_vertices=int(1e5)
    )
    path_2, _ = planner.plan(
        start_state,
        goal_config,
        max_planning_time=50,
        time_horizon=60
    )


    # planner = ModifiedRRTPlannerTimeVarying(
    #     world,
    #     local_planner_nr_coll_steps=2,
    #     local_planner_max_distance=np.inf,
    #     max_nr_vertices=int(1e5)
    # )
    #
    # path_3, _ = planner.plan(
    #     start_state,
    #     goal_config,
    #     max_planning_time=60,
    #     min_planning_time=59
    # )

    def path_len(p):
        return np.linalg.norm(p[1:, :] - p[:-1, :], axis=1).sum()

    # print(path_2)
    path = path_2
    # print(path_len(path_2), path_len(path_3))# 35.29198395465184

    # TODO: RRTConnect!
    # state_start, config_goal, max_planning_time = np.inf
    # print(path, goal_config, world.robot.goal_state)
    # print(goal_config)
    # import matplotlib.pyplot as plt
    # import trimesh
    # import pyrb
    # mesh = trimesh.creation.cylinder(0.1, height=planner.time_horizon)
    # offset = np.append(goal_config, planner.time_horizon/2)
    # mesh.apply_transform(pyrb.kin.rot_trans_to_SE3(p=offset))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # i = planner.vert_cnt
    #
    # # ax.scatter(planner.vertices[:i, 0], planner.vertices[:i, 1], planner.vertices[:i, 2])
    #
    # for i_parent, indxs_children in planner.edges_parent_to_children.items():
    #     for i_child in indxs_children:
    #         q = np.stack([planner.vertices[i_parent], planner.vertices[i_child]], axis=0)
    #         ax.plot(q[:, 0], q[:, 1], q[:, 2], ls="-", marker=".", color="black")
    # ax.plot_trisurf(
    #     mesh.vertices[:, 0],
    #     mesh.vertices[:, 1],
    #     triangles=mesh.faces,
    #     Z=mesh.vertices[:, 2],
    #     color="red",
    #     alpha=0.1
    # )
    # plt.show()
    # print(status.status, status.time_taken, status.nr_verts)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    if path.size:
        for i, state in enumerate(path):
            config = state[:-1]
            t = state[-1]
            world.robot.set_config(config)
            world.set_time(t)
            print(world.get_current_config_smallest_obstacle_distance())
            world.render_world(ax1)
            sub_path = path[:i + 1, :-1]
            world.render_configuration_space(ax2, path=sub_path)
            curr_verts = planner.vertices[:, -1] == t
            if curr_verts.any():
                ax2.scatter(planner.vertices[curr_verts, 0], planner.vertices[curr_verts, 1])
            fig.suptitle(f"Time {t}")
            plt.pause(0.1)
            ax1.cla()
            ax2.cla()
    else:
        for t in range(600):
            world.robot.set_config(np.array([np.pi / 2, 0]))
            world.set_time(t)
            world.render_world(ax1)
            world.render_configuration_space(ax2)
            curr_verts = (planner.vertices[:, -1] - t) == 0
            if curr_verts.any():
                ax2.scatter(planner.vertices[curr_verts, 0], planner.vertices[curr_verts, 1])
            plt.pause(0.01)
            ax1.cla()
            ax2.cla()
            fig.suptitle(f"Time {t} distance: {world.get_current_config_smallest_obstacle_distance():.2f}")
