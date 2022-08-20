import numpy as np

from pyrb.mp.planners.static.local_planners import LocalPlannerStatus


class Tree:

    def __init__(self, max_nr_vertices, vertex_dim):
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

    def append_vertex(self, state, i_parent, edge_cost=None):
        i_new = self.vert_cnt
        self.vertices[i_new, :] = state
        if edge_cost is None:
            edge_cost = np.linalg.norm(state - self.vertices[i_parent, :])
        self.create_edge(i_parent, self.vert_cnt, edge_cost=edge_cost)
        self.vert_cnt += 1
        return i_new

    def create_edge(self, i_parent, i_child, edge_cost=1):
        self.edges_child_to_parent[i_child] = i_parent
        self.set_cost_from_parent(i_parent=i_parent, i_child=i_child, edge_cost=edge_cost)

    def insert_vertex_in_tree(self, i, state):
        self.vertices[i, :] = state

    def find_nearest_vertex(self, state):
        distance = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        i_vert = np.argmin(distance)
        return i_vert, self.vertices[i_vert]

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
        if i_vert == 0:
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
        if i_orphans.size == 0:
            return
        i_vert = i_orphans[0]
        if i_vert == 0:
            self.prune_orphans(i_orphans[1:])
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
        i_orphans = np.append(i_orphans[1:], i_orphans_new)
        self.prune_orphans(i_orphans)


class TreeRewire(Tree):

    def __init__(self, *args, local_planner, nearest_radius, **kwargs):
        super().__init__(*args, **kwargs)
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
            status, path = self.local_planner.plan(
                state_src=state_nearest,
                state_dst=state_new,
                state_global_goal=None,
                full_plan=True
            )
            successful_plan = status == LocalPlannerStatus.REACHED
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
            self.create_edge(i_parent=i_new, i_child=i, edge_cost=edge_cost)

    def get_nearest_vertices_indices(self, state):
        distances = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        return (distances < self.nearest_radius).nonzero()[0]

