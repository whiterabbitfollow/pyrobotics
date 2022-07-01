import numpy as np


from pyrb.mp.planners.rrt import RRTPlanner


class RRTStarPlanner(RRTPlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_to_verts = np.zeros(self.max_nr_vertices)

    def get_nearest_vertices_indices(self, q):
        distances = np.linalg.norm(self.vertices[:self.vert_cnt] - q, axis=1)
        distance_tol = .2
        return np.where(distances < distance_tol),

    def plan(self, q_start, q_goal):
        self.add_vertex_to_tree(q_start)
        path = []
        while not self.is_tree_full():
            q_free = self.sample_collision_free_config()
            i_vert, q_nearest = self.find_nearest_vertex(q_free)
            _, q_new = self.plan_locally(q_nearest, q_free)
            if q_new is not None:
                indxs_qs_nearest = self.get_nearest_vertices_indices(q_new)
                best_indx = self.check_nodes_nearest(indxs_qs_nearest, q_new) or i_vert
                best_edge_cost = np.linalg.norm(self.vertices[best_indx] - q_new)
                self.insert_vertex_in_tree(self.vert_cnt, q_new)
                self.create_edge(i_vert, self.vert_cnt)
                self.set_cost_from_parent(i_vert, self.vert_cnt, best_edge_cost)
                self.vert_cnt += 1

                for indx_q_nearest in indxs_qs_nearest:
                    pass

                if self.is_vertex_in_goal_region(q_new, q_goal):
                    path = self.find_path(q_start, q_goal)
                    break
        return path

    def check_nodes_nearest(self, indxs_qs_nearest, q_new):
        best_cost = -np.inf
        best_indx = None
        for indx_q_nearest in indxs_qs_nearest:
            q_nearest = self.vertices[indx_q_nearest]
            coll_free, _ = self.plan_locally(q_src=q_nearest, q_dst=q_new)
            edge_cost = np.linalg.norm(q_nearest - q_new)
            total_cost_to_new_through_nearest = self.cost_to_verts[indx_q_nearest] + edge_cost
            if total_cost_to_new_through_nearest < best_cost:
                best_cost = total_cost_to_new_through_nearest
                best_indx = indx_q_nearest
        return best_indx

    def set_cost_from_parent(self, i_parent, i_child, edge_cost):
        self.cost_to_verts[i_child] = self.cost_to_verts[i_parent] + edge_cost
