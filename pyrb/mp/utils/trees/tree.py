import numpy as np


class Tree:

    def __init__(self, max_nr_vertices, vertex_dim, space=None):
        self.space = space

        self.cost_to_verts = np.zeros(max_nr_vertices)
        self.edge_costs = np.zeros(max_nr_vertices)

        self.vertices = np.zeros((max_nr_vertices, vertex_dim))
        self.max_nr_vertices = max_nr_vertices
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.edges_parent_to_children = []
        self.vert_cnt = 0
        self.vertices_dict = {}

    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.cost_to_verts.fill(0)
        self.edge_costs.fill(0)
        self.vertices_dict.clear()
        self.edges_parent_to_children.clear()
        self.vert_cnt = 0

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost

    def is_full(self):
        return self.vert_cnt >= self.max_nr_vertices

    def set_root_vertex(self, vertex):
        self.vertices[0, :] = vertex
        self.vertices_dict[tuple(vertex)] = 0
        self.edges_parent_to_children.append([])
        self.vert_cnt += 1

    def add_vertex(self, vertex):
        # TODO: should deprecate this
        self.vertices[self.vert_cnt, :] = vertex
        self.vertices_dict[tuple(vertex)] = self.vert_cnt
        self.edges_parent_to_children.append([])
        self.vert_cnt += 1

    def append_vertex(self, state, i_parent, edge_cost):
        i_old = self.vertices_dict.get(tuple(state))
        if i_old is None:
            i_new = self.vert_cnt
            self.vertices_dict[tuple(state)] = self.vert_cnt
            self.vertices[i_new, :] = state
            self.edges_parent_to_children.append([])
            self.vert_cnt += 1
        else:
            i_new = i_old
        self.create_edge(i_parent, i_new, edge_cost=edge_cost)
        return i_new

    def vertex_exists(self, vertex):
        return tuple(vertex) in self.vertices_dict

    def create_edge(self, i_parent, i_child, edge_cost):
        self.edges_child_to_parent[i_child] = i_parent
        self.set_cost_from_parent(i_parent=i_parent, i_child=i_child, edge_cost=edge_cost)
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edge_costs[i_child] = edge_cost

    def rewire_edge(self, i_parent, i_child, edge_cost):
        i_parent_old = self.edges_child_to_parent[i_child]
        if i_parent_old == i_parent:
            return
        self.edges_parent_to_children[i_parent_old].remove(i_child)
        self.create_edge(i_parent, i_child, edge_cost)
        self.propagate_cost_from_index(i_child)

    def propagate_cost_from_index(self, i_start):
        indxs_children = list(self.edges_parent_to_children[i_start])
        while indxs_children:
            i_child = indxs_children.pop(0)
            i_parent = self.edges_child_to_parent[i_child]
            self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + self.edge_costs[i_child]
            indxs_children.extend(self.edges_parent_to_children[i_child])

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
