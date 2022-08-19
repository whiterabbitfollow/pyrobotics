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

    def insert_vertex_in_tree(self, i, state):
        self.vertices[i, :] = state

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


class TreeRewire(Tree):

    def __init__(self, *args, local_planner, nearest_radius, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_to_verts = np.zeros(self.max_nr_vertices)
        self.nearest_radius = nearest_radius
        self.local_planner = local_planner

    def rewire_nearest(self, i_nearest, state_new):
        i_new = self.vert_cnt
        indxs_states_nearest_coll_free = self.get_collision_free_nearest_indices(state_new)
        indxs_states_all_coll_free = np.append(indxs_states_nearest_coll_free, i_nearest)  # TODO: Not sure this is needed
        self.wire_new_through_nearest(state_new, indxs_states_all_coll_free)
        self.rewire_nearest_through_new(i_new, state_new, indxs_states_nearest_coll_free)

    def wire_new_through_nearest(self, state_new, indxs_states_all_coll_free):
        best_indx, _ = self.find_nearest_indx_with_shortest_path(indxs_states_all_coll_free, state_new)
        self.append_vertex(state_new, i_parent=best_indx)

    def get_collision_free_nearest_indices(self, state_new):
        indxs_states_nearest = self.get_nearest_vertices_indices(state_new)
        indxs_states_nearest_mask = []
        for indx_state_nearest in indxs_states_nearest:
            state_nearest = self.vertices[indx_state_nearest].ravel()
            path = self.local_planner.plan(
                state_src=state_nearest,
                state_dst=state_new,
                state_global_goal=None,
                full_plan=True
            )
            # path_len = path.shape[0]
            # i_parent = indx_state_nearest
            # successful_plan = False
            # for i, state in enumerate(path, 1):
            #     if i == path_len and (state == state_new).all():
            #         successful_plan = True
            #         break
            #     i_child = self.vert_cnt
            #     self.append_vertex(state, i_parent=i_parent)
            #     i_parent = i_child
            successful_plan = path.shape[0] and (path[-1] == state_new).all()
            indxs_states_nearest_mask.append(successful_plan)
        return indxs_states_nearest[indxs_states_nearest_mask]

    def find_nearest_indx_with_shortest_path(self, indxs_states_nearest_coll_free, state_new):
        state_nearest = self.vertices[indxs_states_nearest_coll_free]
        edge_costs = np.linalg.norm(state_nearest - state_new, axis=1)
        total_cost_to_new_through_nearest = self.cost_to_verts[indxs_states_nearest_coll_free] + edge_costs
        best_indx_in_subset = np.argmin(total_cost_to_new_through_nearest)
        best_indx = indxs_states_nearest_coll_free[best_indx_in_subset]
        best_edge_cost = edge_costs[best_indx_in_subset]
        return best_indx, best_edge_cost

    def rewire_nearest_through_new(self, i_new, state_new, indxs_states_nearest_coll_free):
        state_nearest = self.vertices[indxs_states_nearest_coll_free]
        edge_costs = np.linalg.norm(state_nearest - state_new, axis=1)
        cost_through_new = self.cost_to_verts[i_new] + edge_costs
        old_costs = self.cost_to_verts[indxs_states_nearest_coll_free]
        indxs_rewire = indxs_states_nearest_coll_free[cost_through_new < old_costs]
        for i, edge_cost in zip(indxs_rewire, edge_costs):
            self.rewire_edge(i_parent=i_new, i_child=i, edge_cost=edge_cost)

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost

    def rewire_edge(self, i_parent, i_child, edge_cost):
        self.prune_childrens_edges(i_child)
        self.create_edge(i_parent, i_child)
        self.set_cost_from_parent(i_parent=i_parent, i_child=i_child, edge_cost=edge_cost)

    def prune_childrens_edges(self, i_child):
        i_childs_parent = self.edges_child_to_parent[i_child]
        childs_parents_childrens = self.edges_parent_to_children[i_childs_parent]
        childs_parents_childrens.remove(i_child)

    def get_nearest_vertices_indices(self, state):
        distances = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        return (distances < self.nearest_radius).nonzero()[0]

    def append_vertex(self, state, i_parent):
        i_new = self.vert_cnt
        super().append_vertex(state, i_parent)
        edge_cost = np.linalg.norm(state - self.vertices[i_parent])
        self.set_cost_from_parent(i_parent=i_parent, i_child=i_new, edge_cost=edge_cost)

