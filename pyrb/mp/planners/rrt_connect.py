import logging

import numpy as np

from pyrb.mp.planners.rrt import LocalPlannerStatus

RRT_CONNECT_PLANNER_LOGGER_NAME = __file__
logger = logging.getLogger(RRT_CONNECT_PLANNER_LOGGER_NAME)


class RRTConnectPlanner:

    def __init__(
            self,
            tree_start,
            tree_goal,
            local_planner,
            space=None
    ):
        self.state_start = None
        self.goal_region = None
        self.space = space
        self.found_path = False
        self.local_planner = local_planner
        self.tree_start = tree_start
        self.tree_goal = tree_goal
        self.tree_a = None
        self.tree_b = None
        self.path = None

    def clear(self):
        self.tree_start.clear()
        self.tree_goal.clear()
        self.tree_a = None
        self.tree_b = None

    def initialize_planner(self, state_start, goal_region):
        self.state_start = state_start
        self.goal_region = goal_region
        self.found_path = False
        self.tree_start.add_vertex(state_start)
        self.tree_goal.add_vertex(goal_region.state)
        self.tree_a = self.tree_start
        self.tree_b = self.tree_goal

    def can_run(self):
        return not self.tree_start.is_full() and not self.tree_goal.is_full()

    def debug(self, iter_cnt):
        self.goal_region.mask_is_within(self.tree_start.get_vertices())
        i = self.get_goal_state_index()
        cost = self.tree_start.cost_to_verts[i] if i is not None else np.inf
        print(f"{iter_cnt}: {self.tree_start.vert_cnt} best cost: {cost}")

    def run(self):
        space = self.space or self.tree_a.space
        state_free = space.sample_collision_free_state()
        i_nearest_a, state_nearest_a = space.find_nearest_state(self.tree_a.get_vertices(), state_free)
        if i_nearest_a is None or state_nearest_a is None:
            return
        status, local_path_a = self.local_planner.plan(
            space,
            state_nearest_a,
            state_free
        )
        if status == LocalPlannerStatus.TRAPPED:
            return
        goal_head_indices_prune = []
        i_parent_a = i_nearest_a
        for state_new_a in local_path_a:
            i_state_new_a = self.tree_a.append_vertex(state_new_a, i_parent=i_parent_a)
            space = self.space or self.tree_b.space
            i_nearest_b, state_nearest_b = space.find_nearest_state(self.tree_b.get_vertices(), state_new_a)
            if i_nearest_b is None or state_nearest_b is None:
                continue
            status, local_path_b = self.local_planner.plan(
                space,
                state_nearest_b,
                state_new_a,
                max_distance=np.inf
            )
            if status == LocalPlannerStatus.REACHED:
                i_parent_b = i_nearest_b
                for state_new_b in local_path_b:
                    i_parent_b = self.tree_b.append_vertex(state_new_b, i_parent=i_parent_b)
                i_state_start, i_state_goal = self.sort_indices(self.tree_a, i_state_new_a, i_parent_b)
                self.ingest_path_from_tree_goal(i_state_start, i_state_goal)
                logger.debug("Connected path")
                self.found_path = True
                goal_head_indices_prune.append(i_state_goal)
            i_parent_a = i_state_new_a
        all_path_indices = []
        for i_prune in goal_head_indices_prune:
            all_path_indices.extend(self.tree_goal.find_path_indices_to_root_from_vertex_index(i_prune))
        all_path_indices = np.unique(all_path_indices)
        self.tree_goal.prune_orphans(all_path_indices)
        self.tree_a, self.tree_b = self.swap_trees(self.tree_a, self.tree_b)

    def get_goal_state_index(self):
        vertices = self.tree_start.get_vertices()
        mask_vertices_goal = self.goal_region.mask_is_within(vertices)
        i = mask_vertices_goal.nonzero()[0][0] if mask_vertices_goal.any() else None
        return i

    def get_path(self):
        state_start, goal_region = self.state_start, self.goal_region
        i = self.get_goal_state_index()
        if i is not None:
            path = self.tree_start.find_path_to_root_from_vertex_index(i)
            path = path[::-1]
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def sort_indices(self, tree_a, i_a, i_b):
        is_tree_a_start_tree = tree_a == self.tree_start
        if is_tree_a_start_tree:
            i_state_start = i_a
            i_state_goal = i_b
        else:
            i_state_start = i_b
            i_state_goal = i_a
        return i_state_start, i_state_goal

    def swap_trees(self, tree_a, tree_b):
        return tree_b, tree_a

    def ingest_path_from_tree_goal(self, i_state_start, i_state_goal):
        path_state_to_goal = self.tree_goal.find_path_to_root_from_vertex_index(i_state_goal)
        i_parent = i_state_start
        for state_new in path_state_to_goal[1:, :]:
            i_parent = self.tree_start.append_vertex(state_new, i_parent=i_parent)

    def connect_trees(self, i_state_start,  i_state_goal):
        path_state_to_start = self.tree_start.find_path_to_root_from_vertex_index(i_state_start)
        path_state_to_goal = self.tree_goal.find_path_to_root_from_vertex_index(i_state_goal)
        return np.vstack([path_state_to_start[::-1, :], path_state_to_goal[1:, :]])

    def get_planning_meta_data(self):
        return {}