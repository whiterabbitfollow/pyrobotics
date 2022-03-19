import numpy as np
from collections import defaultdict


class RRTPlanner:

    def __init__(self, world, max_nr_vertices=int(1e4), max_distance_local_planner=0.5):
        self.vertices = np.zeros((max_nr_vertices, world.robot.nr_joints))
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.edges_parent_to_children = defaultdict(list)
        self.max_distance_local_planner = max_distance_local_planner
        self.min_step_size = 0.01
        self.vert_cnt = 0
        self.goal_region_radius = 1e-1
        self.max_nr_vertices = max_nr_vertices
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()

    def plan(self, q_start, q_goal):
        self.add_vertex_to_tree(q_start)
        path = []
        while not self.is_tree_full():
            q_free = self.sample_collision_free_config()
            i_vert, q_nearest = self.find_nearest_vertex(q_free)
            q_new = self.plan_locally(q_nearest, q_free)
            if q_new is not None:
                self.insert_vertex_in_tree(self.vert_cnt, q_new)
                self.create_edge(i_vert, self.vert_cnt)
                self.vert_cnt += 1
                if self.is_vertex_in_goal_region(q_new, q_goal):
                    path = self.find_path(q_start, q_goal)
                    break
        return path

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
            if self.world.is_collision_free(q):
                return q

    def find_nearest_vertex(self, q):
        distance = np.linalg.norm(self.vertices[:self.vert_cnt] - q, axis=1)
        i_vert = np.argmin(distance)
        return i_vert, self.vertices[i_vert]

    def insert_vertex_in_tree(self, i, q):
        self.vertices[i, :] = q

    def create_edge(self, i_parent, i_child):
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edges_child_to_parent[i_child] = i_parent

    def add_vertex_to_tree(self, q):
        self.vertices[self.vert_cnt, :] = q
        self.vert_cnt += 1

    def plan_locally(self, q_src, q_dst):
        # assumes q_src is collision free
        q_delta = q_dst - q_src
        distance = np.linalg.norm(q_delta)
        max_nr_steps = int(distance / self.min_step_size)
        if distance > self.max_distance_local_planner:
            nr_steps = int(self.max_distance_local_planner / self.min_step_size)
        else:
            nr_steps = int(distance / self.min_step_size)
        q_new = None
        for i in range(1, nr_steps + 1):
            alpha = i/max_nr_steps
            q = q_dst * alpha + (1-alpha) * q_src
            if self.world.is_collision_free(q):
                q_new = q
            else:
                break
        return q_new
