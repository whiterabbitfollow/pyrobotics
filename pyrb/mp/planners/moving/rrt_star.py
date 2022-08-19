import logging
import time

import numpy as np

from pyrb.mp.planners.moving.rrt import RRTPlannerTimeVarying
from pyrb.mp.planners.utils import start_timer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStarPlannerTimeVarying(RRTPlannerTimeVarying):

    def __init__(self, *args, nearest_radius=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_to_verts = np.zeros(self.max_nr_vertices)
        self.nearest_radius = nearest_radius
        self.ingested_vertices = []
        # TODO: add a list of newest vertices that we can evaluate at the end if they are in goal
        # TODO: could add a cone around sampling space parameterized by time.

    def clear(self):
        super().clear()
        self.cost_to_verts.fill(0)
        self.ingested_vertices.clear()

    def plan(
            self,
            state_start,
            config_goal,
            max_planning_time=np.inf,
            min_planning_time=0,
            time_horizon=300
    ):
        self.state_goal = config_goal
        self.add_vertex_to_tree(state_start)
        time_s, time_elapsed = start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time:
            # TODO: Need some better stopping criteria, that is configurable
            state_free = self.sample_collision_free_config(time_horizon)
            i_nearest, state_nearest = self.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, config_goal)
            if local_path.size:
                self.ingested_vertices.clear()
                for state_new in local_path:
                    # TODO: do we need to rewire every state in the local_path???
                    i_new = self.vert_cnt
                    self.add_vertex_to_tree(state_new)
                    self.ingested_vertices.append(state_new)
                    self.rewire(i_new, state_new)
                ingested_verts = np.vstack(self.ingested_vertices)
                ingested_configs = ingested_verts[:, :-1]
                ingested_times = ingested_verts[:, -1]
                distance = np.linalg.norm(ingested_configs - config_goal, axis=1)
                in_goal = distance < self.goal_region_radius
                if in_goal.any():
                    time_horizon = min(time_horizon, ingested_times[in_goal].min())
                    logger.debug(
                        f"Found solution, "
                        f"time horizon: {time_horizon}, "
                        f"planning time left: {max_planning_time - time_elapsed}"
                    )
            time_elapsed = time.time() - time_s
        path = self.find_path(state_start, config_goal)
        return path, self.compile_planning_data(path, time_elapsed, self.vert_cnt)

    def rewire(self, i_new, state_new):
        self.rewire_past_states(i_new, state_new)
        self.rewire_future_states(i_new, state_new)

    def rewire_past_states(self, i_new, state_new):
        # i_nearest is a bit redundant here...
        indxs_states_nearest_coll_free, local_paths = self.get_collision_free_past_nearest_indices_and_paths(state_new)
        config_new = state_new[:-1]
        best_subset_indx, best_vert_indx, best_edge_cost = self.find_nearest_indx_with_shortest_path(
            indxs_states_nearest_coll_free,
            config_new
        )
        # TODO: could be indx out of bounce here...
        best_local_path = local_paths[best_subset_indx]
        # TODO: should I ingest the other paths as well? ...
        i_parent = self.ingest_local_path(best_local_path, i_start=best_vert_indx)
        self.create_edge(i_parent=i_parent, i_child=i_new)
        edge_cost = np.linalg.norm(self.vertices[i_parent, :-1] - self.vertices[i_new, :-1])
        self.set_cost_from_parent(i_parent=i_parent, i_child=i_new, edge_cost=edge_cost)
        return best_local_path

    def get_collision_free_past_nearest_indices_and_paths(self, state_new):
        """
        Returns the closest nearest vertex indices, which a local planner is able to steer to.
        local path doesn't include the src node
        """
        indxs_states_nearest = self.get_nearest_past_vertices_indices(state_new)
        indxs_states_nearest_mask, local_paths = [], []
        for indx_state_nearest in indxs_states_nearest:
            # Here we need to employ the local planner... we could find the goal state here...
            # and we also produce new plans that we could add to the graph.... should leverage this...
            state_nearest = self.vertices[indx_state_nearest].ravel()
            local_path = self.local_planner.plan(
                state_src=state_nearest,
                state_dst=state_new,
                config_global_goal=None,
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
            successful_plan = local_path.shape[0] > 0 and (local_path[-1] == state_new).all()
            indxs_states_nearest_mask.append(successful_plan)
            if successful_plan:
                local_path_without_target = local_path[:-1]
                local_paths.append(local_path_without_target)
            # TODO: Ingest data here...
            # TODO: need to save plan here.... so that we can ingest it....
        if not np.array(indxs_states_nearest_mask).any():
            print("apa")    # TODO: check why this happens...
        return indxs_states_nearest[indxs_states_nearest_mask], local_paths

    def get_nearest_past_vertices_indices(self, state):
        t = int(state[-1])
        config = state[:-1]
        nr_time_steps_max_act = int(self.nearest_radius/self.max_actuation)
        valid_time_verts_mask = (self.vertices[:self.vert_cnt, -1] < t) \
                                & \
                                (self.vertices[:self.vert_cnt, -1] > (t - nr_time_steps_max_act))
        valid_time_n_distance_verts_indxs = np.array([], dtype=int)
        if valid_time_verts_mask.any():
            valid_time_verts_indxs = valid_time_verts_mask.nonzero()[0]
            distances = np.linalg.norm(self.vertices[:self.vert_cnt, :-1][valid_time_verts_mask] - config, axis=1)
            valid_time_n_distance_verts_indxs = valid_time_verts_indxs[(distances < self.nearest_radius)]
        return valid_time_n_distance_verts_indxs

    def find_nearest_indx_with_shortest_path(self, indxs_states_nearest_coll_free, config_new):
        config_nearest = self.vertices[indxs_states_nearest_coll_free, :-1]
        edge_costs = np.linalg.norm(config_nearest - config_new, axis=1)
        total_cost_to_new_through_nearest = self.cost_to_verts[indxs_states_nearest_coll_free] + edge_costs
        best_indx_in_subset = np.argmin(total_cost_to_new_through_nearest)
        best_indx = indxs_states_nearest_coll_free[best_indx_in_subset]
        best_edge_cost = edge_costs[best_indx_in_subset]
        return best_indx_in_subset, best_indx, best_edge_cost

    def ingest_local_path(self, local_path, i_start):
        i_parent = i_start
        # TODO: start and end should exist in graph...
        for state in local_path:
            i_child = self.vert_cnt
            self.add_vertex_to_tree(state)
            self.ingested_vertices.append(state)
            self.create_edge(i_parent=i_parent, i_child=i_child)
            edge_cost = np.linalg.norm(self.vertices[i_parent, :-1] - self.vertices[i_child, :-1])
            self.set_cost_from_parent(i_parent=i_parent, i_child=i_child, edge_cost=edge_cost)
            i_parent = i_child
        return i_parent

    def rewire_future_states(self, i_new, state_new):
        indxs_states_nearest_coll_free, local_paths = self.get_collision_free_future_nearest_indices_and_paths(state_new)
        # TODO: need to remove the one that I've added?
        # Does not work!!!???
        config_nearest = self.vertices[indxs_states_nearest_coll_free, :-1]
        edge_costs = np.linalg.norm(config_nearest - state_new[:-1], axis=1)
        cost_through_new = self.cost_to_verts[i_new] + edge_costs
        old_costs = self.cost_to_verts[indxs_states_nearest_coll_free]
        sub_indxs_rewire = np.nonzero(cost_through_new < old_costs)[0]
        indxs_rewire = indxs_states_nearest_coll_free[sub_indxs_rewire]
        for i, i_child, edge_cost in zip(sub_indxs_rewire, indxs_rewire, edge_costs):
            local_path_without_target = local_paths[i][:-1]
            self.ingest_local_path(local_path=local_path_without_target, i_start=i_new)
            i_last_local_path = self.vert_cnt
            self.rewire_edge(i_parent_new=i_last_local_path, i_child=i_child)
            self.set_cost_from_parent(i_parent=i_last_local_path, i_child=i_child, edge_cost=edge_cost)

    def get_collision_free_future_nearest_indices_and_paths(self, state_new):
        """
        Returns the closest nearest vertex indices, which a local planner is able to steer to.
        local path doesn't include the src node
        """
        indxs_states_nearest = self.get_nearest_future_vertices_indices(state_new)
        indxs_states_nearest_mask, local_paths = [], []
        for indx_state_nearest in indxs_states_nearest:
            # Here we need to employ the local planner... we could find the goal state here...
            # and we also produce new plans that we could add to the graph.... should leverage this...
            state_nearest = self.vertices[indx_state_nearest].ravel()
            local_path = self.local_planner.plan(
                state_src=state_new,
                state_dst=state_nearest,
                config_global_goal=None,
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
            successful_plan = local_path.shape[0] > 0 and (local_path[-1] == state_new).all()
            indxs_states_nearest_mask.append(successful_plan)
            if successful_plan:
                local_paths.append(local_path)
            # TODO: Ingest data here...
            # TODO: need to save plan here.... so that we can ingest it....
        return indxs_states_nearest[indxs_states_nearest_mask], local_paths

    def get_nearest_future_vertices_indices(self, state):
        t = int(state[-1])
        config = state[:-1]
        nr_time_steps_max_act = int(self.nearest_radius / self.max_actuation)
        valid_time_verts_mask = (self.vertices[:self.vert_cnt, -1] > t) \
                                & \
                                (self.vertices[:self.vert_cnt, -1] < (t + nr_time_steps_max_act))
        valid_time_n_distance_verts_indxs = np.array([], dtype=int)
        if valid_time_verts_mask.any():
            valid_time_verts_indxs = valid_time_verts_mask.nonzero()[0]
            distances = np.linalg.norm(self.vertices[:self.vert_cnt, :-1][valid_time_verts_mask] - config, axis=1)
            valid_time_n_distance_verts_indxs = valid_time_verts_indxs[(distances < self.nearest_radius)]
        return valid_time_n_distance_verts_indxs

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost

    def rewire_edge(self, i_parent_new, i_child):
        self.prune_childrens_edges(i_child)
        self.create_edge(i_parent_new, i_child)

    def prune_childrens_edges(self, i_child):
        i_childs_parent = self.edges_child_to_parent[i_child]
        childs_parents_childrens = self.edges_parent_to_children[i_childs_parent]
        childs_parents_childrens.remove(i_child)

