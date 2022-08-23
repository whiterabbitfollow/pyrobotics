import numpy as np


from pyrb.mp.utils.constants import LocalPlannerStatus
from pyrb.mp.utils.trees.tree import Tree


class TreeRewire(Tree):

    def __init__(self, *args, local_planner, space, nearest_radius, **kwargs):
        super().__init__(*args, **kwargs, vertex_dim=space.dim)
        self.nearest_radius = nearest_radius
        self.space = space
        self.local_planner = local_planner

    def append_vertex_without_rewiring(self, state, i_parent, edge_cost=None):
        super().append_vertex(state, i_parent, edge_cost=edge_cost)

    def append_vertex(self, state, i_parent, edge_cost=None):
        i_new = self.vert_cnt
        i_nearest = i_parent
        indxs_states_nearest = self.space.get_nearest_states_indices(
            states=self.get_vertices(), state=state, nearest_radius=self.nearest_radius
        )
        indxs_states_nearest_coll_free = self.get_collision_free_nearest_indices(state, indxs_states_nearest)
        indxs_states_all_coll_free = np.append(indxs_states_nearest_coll_free, i_nearest)  # TODO: Not sure this is needed
        self.wire_new_through_nearest(state, indxs_states_all_coll_free)
        self.rewire_nearest_through_new(i_new, state, indxs_states_nearest_coll_free)
        return i_new

    def get_collision_free_nearest_indices(self, state_new, indxs_states_nearest):
        indxs_states_nearest_mask = []
        for indx_state_nearest in indxs_states_nearest:
            state_nearest = self.vertices[indx_state_nearest].ravel()
            status, path = self.local_planner.plan(
                space=self.space,
                state_src=state_nearest,
                state_dst=state_new,
                max_distance=np.inf
            )
            successful_plan = status == LocalPlannerStatus.REACHED
            indxs_states_nearest_mask.append(successful_plan)
        return indxs_states_nearest[indxs_states_nearest_mask]

    def wire_new_through_nearest(self, state_new, indxs_states_all_coll_free):
        best_indx, best_edge_cost = self.find_nearest_indx_with_shortest_path(indxs_states_all_coll_free, state_new)
        super().append_vertex(state_new, i_parent=best_indx, edge_cost=best_edge_cost)

    def find_nearest_indx_with_shortest_path(self, indxs_states_nearest_coll_free, state_new):
        states_nearest = self.vertices[indxs_states_nearest_coll_free]
        edge_costs = self.space.distances(state_new, states=states_nearest)
        total_cost_to_new_through_nearest = self.cost_to_verts[indxs_states_nearest_coll_free] + edge_costs
        best_indx_in_subset = np.argmin(total_cost_to_new_through_nearest)
        best_indx = indxs_states_nearest_coll_free[best_indx_in_subset]
        best_edge_cost = edge_costs[best_indx_in_subset]
        return best_indx, best_edge_cost

    def rewire_nearest_through_new(self, i_new, state_new, indxs_states_nearest_coll_free):
        states_nearest = self.vertices[indxs_states_nearest_coll_free]
        edge_costs = self.space.distances(state_new, states=states_nearest)
        cost_through_new = self.cost_to_verts[i_new] + edge_costs
        old_costs = self.cost_to_verts[indxs_states_nearest_coll_free]
        indxs_rewire = indxs_states_nearest_coll_free[cost_through_new < old_costs]
        for i, edge_cost in zip(indxs_rewire, edge_costs):
            self.create_edge(i_parent=i_new, i_child=i, edge_cost=edge_cost)


class TreeRewireSpaceTime(TreeRewire):

    def __init__(self, *args, nearest_time_window, **kwargs):
        self.nearest_time_window = nearest_time_window
        super().__init__(*args, **kwargs)

    def append_vertex(self, state, i_parent, edge_cost=None):
        i_new = self.vert_cnt
        indxs_past = self.get_nearest_past_states_indices(state)
        indxs_past_coll_free = self.get_collision_free_past_nearest_indices(state, indxs_past)
        indxs_past_coll_free = np.append(indxs_past_coll_free, i_parent)
        self.wire_new_through_nearest(state, indxs_past_coll_free)

        indxs_future = self.get_nearest_future_states_indices(state)
        indxs_future_coll_free = self.get_collision_free_future_nearest_indices(state, indxs_future)
        self.rewire_nearest_through_new(i_new, state, indxs_future_coll_free)
        return i_new

    def get_collision_free_past_nearest_indices(self, state_new, indxs_states_nearest):
        indxs_states_nearest_mask = []
        for indx_state_nearest in indxs_states_nearest:
            state_nearest = self.vertices[indx_state_nearest].ravel()
            status, path = self.local_planner.plan(
                space=self.space,
                state_src=state_nearest,
                state_dst=state_new,
                max_distance=np.inf
            )
            successful_plan = status == LocalPlannerStatus.REACHED
            indxs_states_nearest_mask.append(successful_plan)
        return indxs_states_nearest[indxs_states_nearest_mask]

    def get_collision_free_future_nearest_indices(self, state_new, indxs_states_nearest):
        indxs_states_nearest_mask = []
        for indx_state_nearest in indxs_states_nearest:
            state_nearest = self.vertices[indx_state_nearest].ravel()
            status, path = self.local_planner.plan(
                space=self.space,
                state_src=state_new,
                state_dst=state_nearest,
                max_distance=np.inf
            )
            successful_plan = status == LocalPlannerStatus.REACHED
            indxs_states_nearest_mask.append(successful_plan)
        return indxs_states_nearest[indxs_states_nearest_mask]

    def get_nearest_past_states_indices(self, state):
        nearest_radius = self.local_planner.max_actuation
        indices = np.array([], dtype=int)
        t = state[-1]
        t_prv = t
        for t_delta in range(1, self.nearest_time_window + 1):
            t_nxt = self.space.detransition(t, dt=t_delta)
            if t_nxt == t_prv:
                break
            indices = np.append(
                indices,
                self.space.get_indices_of_states_within_time(
                    config=state[:-1],
                    states=self.get_vertices(),
                    t=t_nxt,
                    nearest_radius=nearest_radius * t_delta
                )
            )
            t_prv = t_nxt
        return indices

    def get_nearest_future_states_indices(self, state):
        nearest_radius = self.local_planner.max_actuation
        indices = np.array([], dtype=int)
        t = state[-1]
        t_prv = t
        for t_delta in range(1, self.nearest_time_window + 1):
            t_nxt = self.space.transition(t, dt=t_delta)
            if t_nxt == t_prv:
                break
            indices = np.append(
                indices,
                self.space.get_indices_of_states_within_time(
                    config=state[:-1],
                    states=self.get_vertices(),
                    t=t_nxt,
                    nearest_radius=nearest_radius * t_delta
                )
            )
            t_prv = t_nxt
        return indices