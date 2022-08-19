import time

import numpy as np

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.static.local_planners import LocalPlanner, LocalPlannerStatus
from pyrb.mp.utils.utils import start_timer, compile_planning_data
from pyrb.mp.utils.tree import Tree


class RRTConnectPlanner:

    def __init__(
            self,
            world: BaseMPWorld,
            max_nr_vertices=int(1e4),
            max_distance_local_planner=0.5
    ):
        self.state_goal = None
        self.max_distance_local_planner = max_distance_local_planner
        self.goal_region_radius = 1e-1
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,
            max_distance=max_distance_local_planner,
            global_goal_region_radius=self.goal_region_radius
        )
        self.tree_start = Tree(max_nr_vertices=max_nr_vertices, vertex_dim=world.robot.nr_joints)
        self.tree_goal = Tree(max_nr_vertices=max_nr_vertices, vertex_dim=world.robot.nr_joints)

    def clear(self):
        self.tree_start.clear()
        self.tree_goal.clear()

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.clear()
        self.state_goal = state_goal
        self.tree_start.add_vertex(state_start)
        self.tree_goal.add_vertex(state_goal)

        path = np.array([]).reshape((0, self.state_goal.size))
        time_s, time_elapsed = start_timer()

        tree_a = self.tree_start
        tree_b = self.tree_goal

        while not self.tree_start.is_full() and time_elapsed < max_planning_time and path.size == 0:
            path = self.rrt_connect(tree_a, tree_b)
            tree_a, tree_b = self.swap_trees(tree_a, tree_b)
            time_elapsed = time.time() - time_s
        return path, compile_planning_data(path, time_elapsed, tree_a.vert_cnt + tree_b.vert_cnt)

    def rrt_connect(self, tree_a, tree_b):
        path = np.array([]).reshape((0, self.state_goal.size))
        state_free = self.sample_collision_free_config()
        i_nearest_a, state_nearest_a = tree_a.find_nearest_vertex(state_free)
        status, local_path = self.local_planner.plan(state_nearest_a, state_free)  # , state_goal)
        if status == LocalPlannerStatus.TRAPPED:
            return path
        state_new_a = local_path[-1]
        i_state_new_a = tree_a.vert_cnt
        tree_a.append_vertex(state_new_a, i_parent=i_nearest_a)
        i_nearest_b, state_nearest_b = tree_b.find_nearest_vertex(state_new_a)
        status, local_path = self.local_planner.plan(state_nearest_b, state_new_a, full_plan=True)
        if status == LocalPlannerStatus.REACHED:
            state_new_b = local_path[-1]
            i_state_new_b = tree_b.vert_cnt
            tree_b.append_vertex(state_new_b, i_parent=i_nearest_b)
            i_state_start, i_state_goal = self.sort_indices(tree_a, i_state_new_a, i_state_new_b)
            path = self.connect_trees(
                i_state_start,
                i_state_goal
            )
        return path

    def sort_indices(self, tree_a, i_a, i_b):
        is_tree_a_start_tree = tree_a == self.tree_start
        if is_tree_a_start_tree:
            i_state_start = i_a
            i_state_goal = i_b
        else:
            i_state_start = i_b
            i_state_goal = i_a
        return i_state_start, i_state_goal

    def swap_trees(self, tree_a, tree_b):
        return tree_b, tree_a

    def connect_trees(self, i_state_start,  i_state_goal):
        path_state_to_start = self.tree_start.find_path_to_root_from_vertex_index(i_state_start)
        path_state_to_goal = self.tree_goal.find_path_to_root_from_vertex_index(i_state_goal)
        return np.vstack([path_state_to_start[::-1, :], path_state_to_goal[1:, :]])

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state
