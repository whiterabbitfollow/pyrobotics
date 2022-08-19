import numpy as np
from collections import defaultdict


class Tree:

    def __init__(self, max_nr_vertices, vertex_dim):
        self.edges_parent_to_children = defaultdict(list)
        self.vertices = np.zeros((max_nr_vertices, vertex_dim))
        self.max_nr_vertices = max_nr_vertices
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.vert_cnt = 0

    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.edges_parent_to_children.clear()
        self.vert_cnt = 0

    def is_full(self):
        return self.vert_cnt >= self.max_nr_vertices

    def add_vertex(self, vertex):
        self.vertices[self.vert_cnt, :] = vertex
        self.vert_cnt += 1

    def append_vertex(self, state, i_parent):
        self.vertices[self.vert_cnt, :] = state
        self.create_edge(i_parent, self.vert_cnt)
        self.vert_cnt += 1

    def create_edge(self, i_parent, i_child):
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edges_child_to_parent[i_child] = i_parent

    def find_nearest_vertex(self, state):
        distance = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        i_vert = np.argmin(distance)
        return i_vert, self.vertices[i_vert]

    def get_vertices(self):
        return self.vertices[:self.vert_cnt]

    def get_vertex_parent_index(self, i_vertex):
        return self.edges_child_to_parent[i_vertex]

    def find_path_to_root_from_vertex_index(self, i_vert):
        vertices = self.get_vertices()
        state = vertices[i_vert, :]
        path = [state]
        while i_vert:
            i_vert = self.get_vertex_parent_index(i_vert)
            state = vertices[i_vert, :]
            path.append(state)
        return np.vstack(path)
