import time

import numpy as np
from collections import defaultdict

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.shared import PlanningStatus


def is_vertex_in_goal_region(q, q_goal, goal_region_radius):
    distance = np.linalg.norm(q - q_goal)
    return distance < goal_region_radius


class LocalPlanner:

    def __init__(self, world, min_step_size, max_distance, global_goal_region_radius):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size
        self.world = world

    def plan(self, q_src, q_dst, q_global_goal):
        # assumes q_src is collision free
        q_delta = q_dst - q_src
        distance = np.linalg.norm(q_delta)
        if distance > self.max_distance:
            nr_steps = int(self.max_distance / self.min_step_size)
        else:
            nr_steps = int(distance / self.min_step_size)
        path, collision_free_transition = [], False
        min_transition_distance = 0.2
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            q = q_dst * alpha + (1 - alpha) * q_src
            collision_free_transition = self.world.is_collision_free_state(q)
            if not collision_free_transition:
                break
            is_in_global_goal = is_vertex_in_goal_region(q, q_global_goal, self.global_goal_region_radius)
            if not path or np.linalg.norm(q - path[-1]) > min_transition_distance or i == nr_steps or is_in_global_goal:
                path.append(q)
            if is_in_global_goal:
                break
        return path

    def is_transition_coll_free(self, q_src, q_dst):
        q_delta = q_dst - q_src
        distance = np.linalg.norm(q_delta)
        nr_steps = int(distance / self.min_step_size)
        collision_free_transition = False
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            q = q_dst * alpha + (1 - alpha) * q_src
            collision_free_transition = self.world.is_collision_free_state(q)
            if not collision_free_transition:
                break
        return collision_free_transition


class RRTPlanner:

    def __init__(self, world: BaseMPWorld, max_nr_vertices=int(1e4), max_distance_local_planner=0.5):
        self.max_nr_vertices = max_nr_vertices
        self.vertices = np.zeros((max_nr_vertices, world.robot.nr_joints))
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.edges_parent_to_children = defaultdict(list)
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

    def plan(self, q_start, q_goal, max_planning_time=np.inf):
        self.add_vertex_to_tree(q_start)
        path = []
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            q_free = self.sample_collision_free_config()
            i_vert, q_nearest = self.find_nearest_vertex(q_free)
            local_path = self.local_planner.plan(q_nearest, q_free, q_goal)
            q_new = local_path[-1] if local_path else None
            if q_new is not None:
                self.insert_vertex_in_tree(self.vert_cnt, q_new)
                self.create_edge(i_vert, self.vert_cnt)
                self.vert_cnt += 1
                if is_vertex_in_goal_region(q_new, q_goal, self.goal_region_radius):
                    path = self.find_path(q_start, q_goal)
            time_elapsed = time.time() - time_s
        return path, PlanningStatus(time_taken=time_elapsed, message="Done")

    def start_timer(self):
        time_s = time.time()
        time_elapsed = time.time() - time_s
        return time_s, time_elapsed

    def is_vertex_in_goal_region(self, q, q_goal):
        distance = np.linalg.norm(q - q_goal)
        return distance < self.goal_region_radius

    def is_tree_full(self):
        return self.vert_cnt >= self.max_nr_vertices

    def find_path(self, q_start, q_goal):
        distances = np.linalg.norm(q_goal - self.vertices[:self.vert_cnt], axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            indx, = np.where(mask_vertices_goal)
            i = indx[0]
            q = self.vertices[i, :]
            path = [q]
            while (q != q_start).any():
                i = self.edges_child_to_parent[i]
                q = self.vertices[i, :]
                path.append(q)
            path.reverse()
        else:
            path = []
        return path

    def sample_collision_free_config(self):
        while True:
            q = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(q):
                return q

    def find_nearest_vertex(self, q):
        distance = np.linalg.norm(self.vertices[:self.vert_cnt] - q, axis=1)
        i_vert = np.argmin(distance)
        return i_vert, self.vertices[i_vert]

    def insert_vertex_in_tree(self, i, q):
        self.vertices[i, :] = q
        # TODO: add append?

    def create_edge(self, i_parent, i_child):
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edges_child_to_parent[i_child] = i_parent

    def add_vertex_to_tree(self, q):
        self.vertices[self.vert_cnt, :] = q
        self.vert_cnt += 1
