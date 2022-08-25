import numpy as np


class Tree:

    def __init__(self, max_nr_vertices, vertex_dim, space=None):
        self.space = space
        self.cost_to_verts = np.zeros(max_nr_vertices)
        self.vertices = np.zeros((max_nr_vertices, vertex_dim))
        self.max_nr_vertices = max_nr_vertices
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.vert_cnt = 0

    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.vert_cnt = 0

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost

    def is_full(self):
        return self.vert_cnt >= self.max_nr_vertices

    def add_vertex(self, vertex):
        self.vertices[self.vert_cnt, :] = vertex
        self.vert_cnt += 1

    def append_vertex(self, state, i_parent, edge_cost):
        # TODO: need something like this, and propagate to children that it has changed..
        # mask = np.all(state == self.vertices[:self.vert_cnt], axis=1)
        # if mask.any():
        #     i_new = mask.nonzero()[0][0]
        #     new_cost = self.cost_to_verts[i_parent] + edge_cost
        #     old_cost = self.cost_to_verts[i_new]
        #     if new_cost > old_cost:
        #         return i_new
        # else:
        i_new = self.vert_cnt
        self.vertices[i_new, :] = state
        self.vert_cnt += 1
        self.create_edge(i_parent, i_new, edge_cost=edge_cost)
        return i_new

    def create_edge(self, i_parent, i_child, edge_cost=1):
        self.edges_child_to_parent[i_child] = i_parent
        self.set_cost_from_parent(i_parent=i_parent, i_child=i_child, edge_cost=edge_cost)

    def insert_vertex_in_tree(self, i, state):
        self.vertices[i, :] = state

    def get_vertices(self):
        return self.vertices[:self.vert_cnt]

    def get_edges(self):
        return self.edges_child_to_parent[:self.vert_cnt]

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

    def find_path_indices_to_root_from_vertex_index(self, i_vert):
        path_indices = [i_vert]
        while i_vert:
            i_vert = self.get_vertex_parent_index(i_vert)
            path_indices.append(i_vert)
        return path_indices

    def prune_vertex(self, i_vert):
        if i_vert == 0 or i_vert >= self.vert_cnt:
            # cannot prune root
            return
        self.vertices[i_vert:self.vert_cnt - 1, :] = self.vertices[i_vert + 1:self.vert_cnt, :]
        # find any node that points to it??
        self.edges_child_to_parent[i_vert:self.vert_cnt - 1] = self.edges_child_to_parent[i_vert + 1:self.vert_cnt]
        self.cost_to_verts[i_vert:self.vert_cnt - 1] = self.cost_to_verts[i_vert + 1:self.vert_cnt]
        mask_orphans = self.edges_child_to_parent[:self.vert_cnt - 1] == i_vert
        self.vertices[self.vert_cnt - 1] = 0
        self.edges_child_to_parent[self.vert_cnt - 1] = 0
        self.cost_to_verts[self.vert_cnt - 1] = 0
        self.vert_cnt -= 1
        mask = self.edges_child_to_parent[:self.vert_cnt] > i_vert
        self.edges_child_to_parent[:self.vert_cnt][mask] -= 1
        self.edges_child_to_parent[:self.vert_cnt][mask_orphans] = -1
        i_orphans = np.nonzero(mask_orphans)[0]
        self.prune_orphans(i_orphans)

    def prune_orphans(self, i_orphans):
        while i_orphans.size:
            i_vert = i_orphans[0]
            if i_vert == 0:
                self.prune_orphans(i_orphans[1:])
                return
            self.vertices[i_vert:self.vert_cnt - 1, :] = self.vertices[i_vert + 1:self.vert_cnt, :]
            # find any node that points to it??
            self.edges_child_to_parent[i_vert:self.vert_cnt - 1] = self.edges_child_to_parent[i_vert + 1:self.vert_cnt]
            self.cost_to_verts[i_vert:self.vert_cnt - 1] = self.cost_to_verts[i_vert + 1:self.vert_cnt]
            mask_orphans = self.edges_child_to_parent[:self.vert_cnt - 1] == i_vert
            self.vertices[self.vert_cnt - 1] = 0
            self.edges_child_to_parent[self.vert_cnt - 1] = 0
            self.cost_to_verts[self.vert_cnt - 1] = 0
            self.vert_cnt -= 1
            mask = self.edges_child_to_parent[:self.vert_cnt] > i_vert
            self.edges_child_to_parent[:self.vert_cnt][mask] -= 1
            self.edges_child_to_parent[:self.vert_cnt][mask_orphans] = -1
            i_orphans[i_orphans > i_vert] -= 1
            i_orphans_new = np.nonzero(mask_orphans)[0]
            i_orphans = np.unique(np.append(i_orphans[1:], i_orphans_new))
