import logging
import time

import numpy as np

from pyrb.mp.planners.rrt import RRTPlanner
from pyrb.mp.planners.shared import PlanningStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStarPlanner(RRTPlanner):

    def __init__(self, *args, nearest_radius=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_to_verts = np.zeros(self.max_nr_vertices)
        self.nearest_radius = nearest_radius

    def get_nearest_vertices_indices(self, q):
        distances = np.linalg.norm(self.vertices[:self.vert_cnt] - q, axis=1)
        return np.where(distances < self.nearest_radius)[0]

    def plan(self, q_start, q_goal, max_planning_time=np.inf):
        self.add_vertex_to_tree(q_start)
        path = []
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            q_free = self.sample_collision_free_config()
            i_nearest, q_nearest = self.find_nearest_vertex(q_free)
            _, q_new = self.plan_locally(q_nearest, q_free)
            if q_new is not None:
                self.rewire(i_nearest, q_new)
                if self.is_vertex_in_goal_region(q_new, q_goal):
                    path = self.find_path(q_start, q_goal)
            time_elapsed = time.time() - time_s
        return path, PlanningStatus(time_taken=time_elapsed, message="Done")

    def rewire(self, i_nearest, q_new):
        indxs_qs_nearest_coll_free = self.get_collision_free_nearest_indices(q_new)
        indxs_qs_all_coll_free = np.append(indxs_qs_nearest_coll_free, i_nearest)
        best_indx, best_edge_cost = self.find_nearest_indx_with_shortest_path(indxs_qs_all_coll_free, q_new)
        i_new = self.vert_cnt
        self.insert_vertex_in_tree(i_new, q_new)
        self.create_edge(i_parent=best_indx, i_child=i_new)
        self.set_cost_from_parent(i_parent=best_indx, i_child=self.vert_cnt, edge_cost=best_edge_cost)
        self.vert_cnt += 1
        self.rewire_nearest_through_new(i_new, q_new, indxs_qs_nearest_coll_free)

    def get_collision_free_nearest_indices(self, q_new):
        indxs_qs_nearest = self.get_nearest_vertices_indices(q_new)
        collision_free_neighs = []
        for indx_q_nearest in indxs_qs_nearest:
            q_nearest = self.vertices[indx_q_nearest]
            coll_free, _ = self.plan_locally(q_src=q_nearest, q_dst=q_new)
            collision_free_neighs.append(coll_free)
        return indxs_qs_nearest[collision_free_neighs]

    def find_nearest_indx_with_shortest_path(self, indxs_qs_nearest_coll_free, q_new):
        q_nearest = self.vertices[indxs_qs_nearest_coll_free]
        edge_costs = np.linalg.norm(q_nearest - q_new, axis=1)
        total_cost_to_new_through_nearest = self.cost_to_verts[indxs_qs_nearest_coll_free] + edge_costs
        best_indx_in_subset = np.argmin(total_cost_to_new_through_nearest)
        best_indx = indxs_qs_nearest_coll_free[best_indx_in_subset]
        best_edge_cost = edge_costs[best_indx_in_subset]
        return best_indx, best_edge_cost

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost

    def rewire_nearest_through_new(self, i_new, q_new, indxs_qs_nearest_coll_free):
        q_nearest = self.vertices[indxs_qs_nearest_coll_free]
        edge_costs = np.linalg.norm(q_nearest - q_new, axis=1)
        cost_through_new = self.cost_to_verts[i_new] + edge_costs
        old_costs = self.cost_to_verts[indxs_qs_nearest_coll_free]
        indxs_rewire = indxs_qs_nearest_coll_free[cost_through_new < old_costs]
        for i, edge_cost in zip(indxs_rewire, edge_costs):
            self.rewire_edge(i_parent=i_new, i_child=i)
            self.set_cost_from_parent(i_parent=i_new, i_child=i, edge_cost=edge_cost)

    def rewire_edge(self, i_parent, i_child):
        self.prune_childrens_edges(i_child)
        self.create_edge(i_parent, i_child)

    def prune_childrens_edges(self, i_child):
        i_childs_parent = self.edges_child_to_parent[i_child]
        childs_parents_childrens = self.edges_parent_to_children[i_childs_parent]
        childs_parents_childrens.remove(i_child)
