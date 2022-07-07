import time

import numpy as np
from collections import defaultdict

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.shared import PlanningData, Status


def is_vertex_in_goal_region(q, q_goal, goal_region_radius):
    distance = np.linalg.norm(q - q_goal)
    return distance < goal_region_radius


class LocalPlanner:

    def __init__(self, world, min_step_size, max_distance, global_goal_region_radius):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size
        self.world = world

    def plan(self, state_src, state_dst, state_global_goal):
        # assumes state_src is collision free
        state_delta = state_dst - state_src
        distance = np.linalg.norm(state_delta)
        nr_steps = int(min(distance, self.max_distance) / self.min_step_size)
        path, collision_free_transition = np.array([]), False
        state_closest = None
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            collision_free_transition = self.world.is_collision_free_state(state)
            is_in_global_goal = is_vertex_in_goal_region(state, state_global_goal, self.global_goal_region_radius)
            if collision_free_transition:
                state_closest = state
            if is_in_global_goal or not collision_free_transition:
                break
        if state_closest is not None:
            min_transition_distance = 0.2
            distance = np.linalg.norm(state_closest - state_src)
            nr_steps = int(distance/min_transition_distance)
            path = np.linspace(state_src, state_closest, nr_steps) if nr_steps > 1 else np.vstack([state_src, state_closest])
            path = path[1:, :]
        return path

    def is_transition_coll_free(self, state_src, state_dst):
        state_delta = state_dst - state_src
        distance = np.linalg.norm(state_delta)
        nr_steps = int(distance / self.min_step_size)
        collision_free_transition = False
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            collision_free_transition = self.world.is_collision_free_state(state)
            if not collision_free_transition:
                break
        return collision_free_transition


class RRTPlanner:

    def __init__(self, world: BaseMPWorld, max_nr_vertices=int(1e4), max_distance_local_planner=0.5):
        self.max_nr_vertices = max_nr_vertices
        self.edges_parent_to_children = defaultdict(list)
        self.vertices = np.zeros((max_nr_vertices, world.robot.nr_joints))
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.max_distance_local_planner = max_distance_local_planner
        self.vert_cnt = 0
        self.goal_region_radius = 1e-1
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,
            max_distance=max_distance_local_planner,
            global_goal_region_radius=self.goal_region_radius
        )

    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.edges_parent_to_children.clear()
        self.vert_cnt = 0

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.clear()
        self.add_vertex_to_tree(state_start)
        path = []
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, state_goal)
            state_new = local_path[-1] if local_path else None
            if state_new is not None:
                self.append_vertex(state_new, i_parent=i_nearest)
                if is_vertex_in_goal_region(state_new, state_goal, self.goal_region_radius):
                    path = self.find_path(state_start, state_goal)
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)

    def start_timer(self):
        time_s = time.time()
        time_elapsed = time.time() - time_s
        return time_s, time_elapsed

    def is_vertex_in_goal_region(self, state, state_goal):
        distance = np.linalg.norm(state - state_goal)
        return distance < self.goal_region_radius

    def is_tree_full(self):
        return self.vert_cnt >= self.max_nr_vertices

    def find_path(self, state_start, state_goal):
        distances = np.linalg.norm(state_goal - self.vertices[:self.vert_cnt], axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            i = mask_vertices_goal.nonzero()[0]
            state = self.vertices[i, :]
            path = [state]
            while (state != state_start).any():
                i = self.edges_child_to_parent[i]
                state = self.vertices[i, :]
                path.append(state)
            path.reverse()
        else:
            path = []
        return path

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state

    def find_nearest_vertex(self, state):
        distance = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        i_vert = np.argmin(distance)
        return i_vert, self.vertices[i_vert]

    def insert_vertex_in_tree(self, i, state):
        self.vertices[i, :] = state

    def append_vertex(self, state, i_parent):
        self.vertices[self.vert_cnt, :] = state
        self.create_edge(i_parent, self.vert_cnt)
        self.vert_cnt += 1

    def create_edge(self, i_parent, i_child):
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edges_child_to_parent[i_child] = i_parent

    def add_vertex_to_tree(self, state):
        self.vertices[self.vert_cnt, :] = state
        self.vert_cnt += 1

    def compile_planning_data(self, path, time_elapsed):
        status = Status.SUCCESS if path else Status.FAILURE
        return PlanningData(status=status, time_taken=time_elapsed)


class RRTPlannerModified(RRTPlanner):

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.add_vertex_to_tree(state_start)
        path = []
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, state_goal)
            i_prev = i_nearest
            for state in local_path:
                self.insert_vertex_in_tree(self.vert_cnt, state)
                self.create_edge(i_parent=i_prev, i_child=self.vert_cnt)
                i_prev = self.vert_cnt
                self.vert_cnt += 1
                if is_vertex_in_goal_region(state, state_goal, self.goal_region_radius):
                    path = self.find_path(state_start, state_goal)
                    break
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)

