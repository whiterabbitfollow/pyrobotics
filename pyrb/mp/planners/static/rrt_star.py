import logging
import time
from collections import defaultdict

import numpy as np

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.shared import Status, PlanningData
from pyrb.mp.planners.static.rrt import LocalPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStarPlanner:

    def __init__(self, world: BaseMPWorld, max_nr_vertices=int(1e4), max_distance_local_planner=0.5, nearest_radius=.2):
        self.max_nr_vertices = max_nr_vertices
        self.max_nr_vertices = max_nr_vertices
        self.state_goal = None
        self.edges_parent_to_children = defaultdict(list)
        self.vertices = np.zeros((max_nr_vertices, world.robot.nr_joints))
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.max_distance_local_planner = max_distance_local_planner
        self.vert_cnt = 0
        self.goal_region_radius = 1e-1
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,
            max_distance=max_distance_local_planner,
            global_goal_region_radius=self.goal_region_radius
        )
        self.cost_to_verts = np.zeros(self.max_nr_vertices)
        self.nearest_radius = nearest_radius

    def start_timer(self):
        time_s = time.time()
        time_elapsed = time.time() - time_s
        return time_s, time_elapsed

    def is_vertex_in_goal_region(self, state):
        distance = np.linalg.norm(state - self.state_goal)
        return distance < self.goal_region_radius

    def is_tree_full(self):
        return self.vert_cnt >= self.max_nr_vertices
    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.edges_parent_to_children.clear()
        self.vert_cnt = 0
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

    def find_path(self, state_start):
        distances = np.linalg.norm(self.state_goal - self.vertices[:self.vert_cnt], axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            i = mask_vertices_goal.nonzero()[0][0]
            state = self.vertices[i, :]
            indxs = [i]
            while (state != state_start).any():
                i = self.edges_child_to_parent[i]
                state = self.vertices[i, :]
                if i in indxs:
                    logging.error("Loop in path")
                    break
                indxs.append(i)
            indxs.reverse()
            path = self.vertices[indxs]
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state

    def find_nearest_vertex(self, state):
        distance = np.linalg.norm(self.vertices[:self.vert_cnt] - state, axis=1)
        i_vert = np.argmin(distance)
        return i_vert, self.vertices[i_vert]

    def insert_vertex_in_tree(self, i, state):
        self.vertices[i, :] = state

    def append_vertex(self, state, i_parent):
        self.vertices[self.vert_cnt, :] = state
        self.create_edge(i_parent, self.vert_cnt)
        self.vert_cnt += 1

    def create_edge(self, i_parent, i_child):
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edges_child_to_parent[i_child] = i_parent

    def add_vertex_to_tree(self, state):
        self.vertices[self.vert_cnt, :] = state
        self.vert_cnt += 1

    def compile_planning_data(self, path, time_elapsed):
        status = Status.SUCCESS if path.size else Status.FAILURE
        return PlanningData(status=status, time_taken=time_elapsed, nr_verts=self.vert_cnt)

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

