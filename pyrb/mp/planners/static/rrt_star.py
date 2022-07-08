import logging
import time

import numpy as np

from pyrb.mp.planners.static.rrt import RRTPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStarPlanner(RRTPlanner):

    def __init__(self, *args, nearest_radius=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_to_verts = np.zeros(self.max_nr_vertices)
        self.nearest_radius = nearest_radius

    def clear(self):
        super().clear()
        self.cost_to_verts.fill(0)

    def get_nearest_vertices_indices(self, state):
        distances = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        return (distances < self.nearest_radius).nonzero()[0]

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.clear()
        self.state_goal = state_goal
        self.add_vertex_to_tree(state_start)
        path = []
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, self.state_goal)
            state_new = local_path[-1] if local_path.size > 0 else None
            if state_new is not None:
                self.rewire(i_nearest, state_new)
                if self.is_vertex_in_goal_region(state_new):
                    logger.debug("Found path to goal!!!")
                    path = self.find_path(state_start)
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)

    def rewire(self, i_nearest, state_new):
        indxs_states_nearest_coll_free = self.get_collision_free_nearest_indices(state_new)
        indxs_states_all_coll_free = np.append(indxs_states_nearest_coll_free, i_nearest)
        best_indx, best_edge_cost = self.find_nearest_indx_with_shortest_path(indxs_states_all_coll_free, state_new)
        i_new = self.vert_cnt
        self.insert_vertex_in_tree(i_new, state_new)
        self.create_edge(i_parent=best_indx, i_child=i_new)
        self.set_cost_from_parent(i_parent=best_indx, i_child=self.vert_cnt, edge_cost=best_edge_cost)
        self.vert_cnt += 1
        self.rewire_nearest_through_new(i_new, state_new, indxs_states_nearest_coll_free)

    def get_collision_free_nearest_indices(self, state_new):
        indxs_states_nearest = self.get_nearest_vertices_indices(state_new)
        indxs_states_nearest_mask = []
        for indx_state_nearest in indxs_states_nearest:
            state_nearest = self.vertices[indx_state_nearest].ravel()
            path = self.local_planner.plan(
                state_src=state_nearest,
                state_dst=state_new,
                state_global_goal=self.state_goal,
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
            self.rewire_edge(i_parent=i_new, i_child=i)
            self.set_cost_from_parent(i_parent=i_new, i_child=i, edge_cost=edge_cost)

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost

    def rewire_edge(self, i_parent, i_child):
        self.prune_childrens_edges(i_child)
        self.create_edge(i_parent, i_child)

    def prune_childrens_edges(self, i_child):
        i_childs_parent = self.edges_child_to_parent[i_child]
        childs_parents_childrens = self.edges_parent_to_children[i_childs_parent]
        childs_parents_childrens.remove(i_child)


class RRTStarPlannerModified(RRTStarPlanner):

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.clear()
        self.state_goal = state_goal
        self.add_vertex_to_tree(state_start)
        path = np.array([]).reshape((-1, ) + state_goal.shape)
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, state_goal)
            for state_new in local_path:
                self.rewire(i_nearest, state_new)
                if self.is_vertex_in_goal_region(state_new):
                    logger.debug("Found path to goal!!!")
                    path = self.find_path(state_start)
                    break
                i_nearest = self.vert_cnt - 1
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)

