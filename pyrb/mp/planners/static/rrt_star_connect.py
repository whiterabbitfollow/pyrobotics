import time

import numpy as np

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.static.local_planners import LocalPlanner, LocalPlannerStatus
from pyrb.mp.utils.utils import start_timer, compile_planning_data
from pyrb.mp.utils.tree import TreeRewire


class RRTStarConnectPlanner:
    # TODO: Not sure about this one....
    def __init__(
            self,
            world: BaseMPWorld,
            max_nr_vertices=int(1e4),
            max_distance_local_planner=0.5,
            nearest_radius=0.2
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
        self.tree_start = TreeRewire(
            max_nr_vertices=max_nr_vertices,
            vertex_dim=world.robot.nr_joints,
            nearest_radius=nearest_radius,
            local_planner=self.local_planner
        )
        self.tree_goal = TreeRewire(
            max_nr_vertices=max_nr_vertices,
            vertex_dim=world.robot.nr_joints,
            nearest_radius=nearest_radius,
            local_planner=self.local_planner
        )

    def clear(self):
        self.tree_start.clear()
        self.tree_goal.clear()

    def plan(
            self,
            state_start,
            state_goal,
            max_planning_time=np.inf,
            min_planning_time=0
    ):
        self.clear()
        self.state_goal = state_goal
        self.tree_start.add_vertex(state_start)
        self.tree_goal.add_vertex(state_goal)
        path = np.array([]).reshape((0, self.state_goal.size))
        time_s, time_elapsed = start_timer()
        tree_a = self.tree_start
        tree_b = self.tree_goal
        found_solution = False
        while (
                not self.tree_start.is_full() and
                time_elapsed < max_planning_time and
                not found_solution
        ) or time_elapsed < min_planning_time:
            trees_connected = self.rrt_connect(tree_a, tree_b)
            tree_a, tree_b = self.swap_trees(tree_a, tree_b)
            time_elapsed = time.time() - time_s
            if not found_solution and trees_connected:
                found_solution = True
        if found_solution:
            path = self.find_path(state_goal)
        return path, compile_planning_data(path, time_elapsed, tree_a.vert_cnt + tree_b.vert_cnt)

    def find_path(self, state_start):
        vertices = self.tree_start.get_vertices()
        distances = np.linalg.norm(self.state_goal - vertices, axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            indices = mask_vertices_goal.nonzero()[0]
            i_min_cost = indices[np.argmin(self.tree_start.cost_to_verts[indices])]
            path = self.tree_start.find_path_to_root_from_vertex_index(i_min_cost)
            path = path[::-1]
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def rrt_connect(self, tree_a, tree_b):
        connected = False
        state_free = self.sample_collision_free_config()
        i_nearest_a, state_nearest_a = tree_a.find_nearest_vertex(state_free)
        status, local_path = self.local_planner.plan(state_nearest_a, state_free)  # , state_goal)
        if status == LocalPlannerStatus.TRAPPED:
            return connected
        state_new_a = local_path[-1]
        i_state_new_a = tree_a.vert_cnt
        tree_a.rewire_nearest(i_nearest_a, state_new_a)
        i_nearest_b, state_nearest_b = tree_b.find_nearest_vertex(state_new_a)
        status, local_path = self.local_planner.plan(state_nearest_b, state_new_a, full_plan=True)
        if status == LocalPlannerStatus.REACHED:
            state_new_b = local_path[-1]
            connected = True
            i_state_new_b = tree_b.vert_cnt
            tree_b.append_vertex(state_new_b, i_parent=i_nearest_b)
            i_state_start, i_state_goal = self.sort_indices(tree_a, i_state_new_a, i_state_new_b)
            # path = self.connect_trees(
            #     i_state_start,
            #     i_state_goal
            # )
            self.ingest_path_from_tree_goal(i_state_start, i_state_goal)
        return connected

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

    def ingest_path_from_tree_goal(self, i_state_start, i_state_goal):
        path_state_to_goal = self.tree_goal.find_path_to_root_from_vertex_index(i_state_goal)
        i_parent = i_state_start
        for state_new in path_state_to_goal[1:, :]:
            i_child = self.tree_start.vert_cnt
            self.tree_start.append_vertex(state_new, i_parent=i_parent)
            i_parent = i_child
            # TODO: prune...

    def connect_trees(self, i_state_start,  i_state_goal):
        path_state_to_start = self.tree_start.find_path_to_root_from_vertex_index(i_state_start)
        path_state_to_goal = self.tree_goal.find_path_to_root_from_vertex_index(i_state_goal)
        return np.vstack([path_state_to_start[::-1, :], path_state_to_goal[1:, :]])

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state
