import time

from pyrb.mp.planners.rrt_connect import RRTConnectPlanner
from pyrb.traj.interpolate import interpolate_single_point_along_line


def start_timer():
    time_s = time.time()
    time_elapsed = time.time() - time_s
    return time_s, time_elapsed


class BaseReRRT:

    def __init__(
            self,
            world,
            rrt_planner,
            goal_region
    ):
        self.world = world
        self.planner = rrt_planner
        self.goal_region = goal_region

    def find_path(self, start_config, goal_config, path_initialize=None, max_planning_time=1):
        time_s, time_elapsed = start_timer()
        self.world.set_start_config(start_config)
        self.goal_region.set_goal_state(goal_config)
        self.planner.clear()
        self.planner.initialize_planner(start_config, self.goal_region)
        if path_initialize is not None and path_initialize.size > 0:
            path_validated = self.validate_path(path_initialize)
            if self.goal_region.is_within(path_validated[-1]):
                return path_validated
            else:
                # re-use validated path somehow
                pass
        iter_cnt = 0
        while (
                self.planner.can_run() and (not self.planner.found_path and time_elapsed < max_planning_time)
        ):
            self.planner.run()
            time_elapsed = time.time() - time_s
            iter_cnt += 1
        path = self.planner.get_path()
        return path

    def validate_path(self, path):
        min_step_size = 0.05
        path_validated = [path[0, :]]
        for config_src, config_dst in zip(path[:-1, :], path[1:, :]):
            dist = np.linalg.norm(config_dst - config_src)
            nr_coll_steps = max(int(dist / min_step_size), 1)
            if self.world.is_collision_free_transition(config_src, config_dst, nr_coll_steps=nr_coll_steps):
                path_validated.append(config_dst)
            else:
                break
        return np.vstack(path_validated)

    def rollout(self,  start_config, goal_config, max_steps=100, max_planning_time_per_step=1):
        trajectory = [self.world.start_config.copy()]
        solved = False
        collision_free = True
        step_nr = 0
        self.world.start_config = start_config
        while not solved and collision_free and step_nr < max_steps:
            path = self.find_path(
                self.world.start_config,
                goal_config,
                max_planning_time=max_planning_time_per_step
            )
            collision_free, solved = self.step_along_path(path)
            trajectory.append(self.world.start_config.copy())
            step_nr += 1
        trajectory = np.vstack(trajectory)
        return solved, trajectory

    def step_along_path(self, path):
        raise NotImplementedError()


class ReRRT(BaseReRRT):

    def __init__(self, *args, max_step_distance, distance_thr, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_step_distance = max_step_distance
        self.distance_thr = distance_thr

    def find_path(self, start_config, goal_config, path_initialize=None, max_planning_time=1):
        path = super().find_path(start_config, goal_config, path_initialize, max_planning_time)
        if path.shape[0] > 2:
            path = post_process_path_continuous_sampling(
                path,
                self.planner.space,
                self.planner.local_planner,
                self.goal_region,
                max_cnt=100,
                max_cnt_no_improvement=0
            )
        return path

    def step_along_path(self, path):
        if path.size:
            delta_config = self.compute_step_along_path(path)
        else:
            delta_config = self.compute_step_no_path_found()
        return self.step_delta_config(delta_config)

    def compute_step_along_path(self, path):
        distance_along_path = compute_max_distance_l_infty_dim_2(
            config_src=path[0, :], config_dst=path[1, :],
            dim_max_distance=self.max_step_distance
        )
        config_nxt = interpolate_single_point_along_line(
            config_src=path[0, :], config_dst=path[1, :],
            distance_to_point=distance_along_path
        )
        delta_config = config_nxt - path[0, :]
        return delta_config

    def compute_step_no_path_found(self):
        nr_joints = world.robot.nr_joints
        world.robot.set_config(world.start_config)
        distance = world.get_current_config_smallest_obstacle_distance()
        if distance < self.distance_thr:
            delta_config_max_distance = self.compute_step_with_max_distance(distance)
        else:
            delta_config_max_distance = np.zeros((nr_joints,))
        return delta_config_max_distance

    def compute_step_with_max_distance(self, distance):
        nr_joints = self.world.robot.nr_joints
        delta_configs = list(product(*([(-max_step_distance, 0, max_step_distance)] * nr_joints)))
        delta_configs.remove(tuple([0] * nr_joints))
        delta_configs = np.array(delta_configs)
        distance_max = distance
        delta_config_max_distance = np.zeros((nr_joints,))
        for delta_config in delta_configs:
            config_nxt = world.start_config + delta_config
            if not self.world.robot.is_configuration_feasible(config_nxt):
                continue
            world.robot.set_config(config_nxt)
            distance = world.get_current_config_smallest_obstacle_distance()
            if distance_max < distance:
                delta_config_max_distance = delta_config
                distance_max = distance
        return delta_config_max_distance

    def step_delta_config(self, delta_config):
        collision_free = self.world.step(delta_config)
        solved = False
        if collision_free:
            solved = self.goal_region.is_within(self.world.start_config)
        return collision_free, solved


if __name__ == "__main__":
    from examples.replanning.agent_and_adv.agent_n_adversary_world import ReplanningAgentAdversary2DWorld
    from pyrb.mp.utils.spaces import RealVectorStateSpace
    from pyrb.mp.utils.trees.tree import Tree
    from pyrb.mp.planners.local_planners import LocalPlanner
    from pyrb.mp.utils.goal_regions import RealVectorGoalRegion
    from pyrb.mp.post_processing.post_processing import post_process_path_continuous_sampling
    import numpy as np
    from itertools import product
    from pyrb.traj.interpolate import compute_max_distance_l_infty_dim_2

    world = ReplanningAgentAdversary2DWorld()

    state_space = RealVectorStateSpace(
        world, world.robot.nr_joints, world.robot.joint_limits
    )
    # planner = RRTPlanner(
    #     space=state_space,
    #     tree=Tree(max_nr_vertices=int(1e5), vertex_dim=state_space.dim),
    #     local_planner=local_planner
    # )
    max_step_distance = 0.1
    min_coll_step_size = 0.025
    planner = RRTConnectPlanner(
        space=state_space,
        tree_start=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
        tree_goal=Tree(max_nr_vertices=int(1e4), vertex_dim=state_space.dim),
        local_planner=LocalPlanner(
            min_coll_step_size=min_coll_step_size,
            max_distance=0.5
        )
    )
    max_cnt = 20,
    max_cnt_no_improvement = 0
    goal_region = RealVectorGoalRegion()
    # post_process_func = partial(
    #     post_process_path_continuous,
    #     space=planner.space,
    #     local_planner=planner.local_planner,
    #     goal_region=goal_region
    # )
    replanner = ReRRT(
        world,
        planner,
        goal_region,
        max_step_distance=max_step_distance,
        distance_thr=0.1
    )
    # world.start_config
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # world.render_world(ax1)
    # world.render_configuration_space(ax2)
    #
    # if path.size:
    #     ax2.plot(path[:, 0], path[:, 1])
    #     ax2.plot(path_pp[:, 0], path_pp[:, 1])
    #
    # plt.show()
    solved_cnt = 0
    path_acc = 0
    step_acc = 0
    cnt = 0
    world.reset(seed=7)
    print(world.obstacles.traj[:10, :])
    print(world.start_config)
    print(world.goal_config)



    # solved, trajectory = replanner.rollout(
    #     start_config=world.start_config,
    #     goal_config=world.goal_config,
    #     max_steps=1000
    # )
    # solved_cnt += solved
    # cnt += 1
    #
    # for pp_flag in (True, False):
    #     tbar = tqdm.tqdm(range(100))
    #     for i in tbar:
    #         try:
    #             world.reset(seed=i)
    #         except:
    #             continue
    #         solved, trajectory = replanner.rollout(
    #             start_config=world.start_config,
    #             goal_config=world.goal_config,
    #             max_steps=100
    #         )
    #         solved_cnt += solved
    #         cnt += 1
    #         if solved:
    #             path_acc += np.linalg.norm(trajectory[1:, :] - trajectory[:-1, :], axis=1).sum()
    #             step_acc += trajectory.shape[0]
    #         tbar.set_description(f"{solved_cnt/cnt} {path_acc/solved_cnt} {step_acc/solved_cnt}")
    # print(solved)
    # fig, ax = plt.subplots(1, 1)
    # ax.add_patch(Circle(tuple(goal_region.state), goal_region.radius, color="red", alpha=0.1))
    # ax.plot(trajectory[:, 0], trajectory[:, 1])
    # ax.set_aspect("equal")
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # path = None
    #
    # for t in range(78, 1_000):
    #     world.set_time(t)
    #     world.render_world(ax1)
    #     path = replanner.find_path(
    #         path_initialize=path,
    #         start_config=world.start_config,
    #         goal_config=world.goal_config,
    #         max_planning_time=1
    #     )
    #     world.render_configuration_space(ax2, path=path)
    #     collision_free, solved = replanner.step_along_path(path)
    #     if path.size:
    #         line = np.vstack([path[0, :], world.start_config])
    #         ax2.plot(line[:, 0], line[:, 1])
    #         path = path[1:, :]
    #         path[0, :] = world.start_config
    #     else:
    #         print(f"Not path found t {t}")
    #     plt.suptitle(f"t: {t}")
    #     plt.pause(0.1)
    #     ax1.cla()
    #     ax2.cla()
    #     if not collision_free:
    #         break
