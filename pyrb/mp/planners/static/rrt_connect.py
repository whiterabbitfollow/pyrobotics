import time

import numpy as np

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.shared import Status, PlanningData
from pyrb.mp.planners.static.rrt import Tree, is_vertex_in_goal_region

from enum import Enum, auto


class LocalPlannerStatus(Enum):
    TRAPPED = auto()
    ADVANCED = auto()
    REACHED = auto()


class LocalPlanner:

    def __init__(self, world, min_step_size, max_distance, global_goal_region_radius):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size
        self.world = world

    def plan(self, state_src, state_dst, state_global_goal=None, full_plan=False):
        # assumes state_src is collision free
        state_delta = state_dst - state_src
        distance = np.linalg.norm(state_delta)
        if not full_plan:
            distance = min(distance, self.max_distance)
        nr_steps = int(distance / self.min_step_size)
        state_closest = None
        status = LocalPlannerStatus.TRAPPED
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            collision_free_transition = self.world.is_collision_free_state(state)
            is_in_global_goal = state_global_goal is not None and is_vertex_in_goal_region(
                state,
                state_global_goal,
                self.global_goal_region_radius
            )
            if collision_free_transition:
                state_closest = state
                status = LocalPlannerStatus.ADVANCED
            else:
                status = LocalPlannerStatus.TRAPPED
                state_closest = None
            if is_in_global_goal or not collision_free_transition:
                break
        if state_closest is not None and np.isclose(state_closest, state_dst).all():
            status = LocalPlannerStatus.REACHED
        return status, state_closest


class RRTConnectPlanner:

    def __init__(self, world: BaseMPWorld, max_nr_vertices=int(1e4), max_distance_local_planner=0.5):
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
        time_s, time_elapsed = self.start_timer()

        tree_a = self.tree_start
        tree_b = self.tree_goal

        while not self.tree_start.is_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest_a, state_nearest_a = tree_a.find_nearest_vertex(state_free)
            status, state_new_a = self.local_planner.plan(state_nearest_a, state_free)      # , state_goal)
            if status == LocalPlannerStatus.TRAPPED:
                time_elapsed = time.time() - time_s
                tree_a, tree_b = self.swap_trees(tree_a, tree_b)
                continue
            i_state_new_a = tree_a.vert_cnt
            tree_a.append_vertex(state_new_a, i_parent=i_nearest_a)
            i_nearest_b, state_nearest_b = tree_b.find_nearest_vertex(state_new_a)
            status, state_new_b = self.local_planner.plan(state_nearest_b, state_new_a, full_plan=True)
            if status == LocalPlannerStatus.REACHED:
                i_state_new_b = tree_b.vert_cnt
                tree_b.append_vertex(state_new_b, i_parent=i_nearest_b)
                i_state_start, i_state_goal = self.sort_indices(tree_a, i_state_new_a, i_state_new_b)
                path = self.connect_trees(
                    i_state_start,
                    i_state_goal
                )
                break
            tree_a, tree_b = self.swap_trees(tree_a, tree_b)
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)

    def compile_planning_data(self, path, time_elapsed):
        status = Status.SUCCESS if path.size else Status.FAILURE
        return PlanningData(
            status=status,
            time_taken=time_elapsed,
            nr_verts=self.tree_start.vert_cnt + self.tree_goal.vert_cnt
        )

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
        path_state_to_start = self.find_path_to_root_from_vertex_index(self.tree_start, i_state_start)
        path_state_to_goal = self.find_path_to_root_from_vertex_index(self.tree_goal, i_state_goal)
        return np.vstack([path_state_to_start[::-1, :], path_state_to_goal[1:, :]])

    def find_path_to_root_from_vertex_index(self, tree, i_vert):
        vertices = tree.get_vertices()
        state = vertices[i_vert, :]
        path = [state]
        while i_vert:
            i_vert = tree.get_vertex_parent_index(i_vert)
            state = vertices[i_vert, :]
            path.append(state)
        return np.vstack(path)

    def start_timer(self):
        time_s = time.time()
        time_elapsed = time.time() - time_s
        return time_s, time_elapsed

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state


