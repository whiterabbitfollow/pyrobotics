import logging

import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus

RRT_PLANNER_LOGGER_NAME = __file__
logger = logging.getLogger(RRT_PLANNER_LOGGER_NAME)


class RRTPlanner:

    def __init__(
            self,
            space,
            tree,
            local_planner
    ):
        self.tree = tree
        self.state_start = None
        self.goal_region = None
        self.space = space
        self.found_path = False
        self.local_planner = local_planner

    def clear(self):
        self.tree.clear()

    def initialize_planner(self, state_start, goal_region):
        self.state_start = state_start
        self.goal_region = goal_region
        self.found_path = False
        self.tree.add_vertex(state_start)

    def can_run(self):
        return not self.tree.is_full()

    def sample(self):
        return self.space.sample_collision_free_state()

    def run(self):
        state_free = self.sample()
        i_nearest, state_nearest = self.space.find_nearest_state(self.tree.get_vertices(), state_free)
        if i_nearest is None or state_nearest is None:
            return
        status, local_path = self.local_planner.plan(
            self.space,
            state_nearest,
            state_free,
            goal_region=self.goal_region
        )
        if status != LocalPlannerStatus.TRAPPED:
            i_parent = i_nearest
            for state_new in local_path:
                edge_cost = self.space.transition_cost(state_new, self.tree.vertices[i_parent])
                if self.can_run() and not self.tree.vertex_exists(state_new):
                    i_new = self.tree.append_vertex(
                        state_new, i_parent=i_parent, edge_cost=edge_cost
                    )
                    if self.goal_region.is_within(state_new):
                        self.report_goal_state(i_new)
                    i_parent = i_new

    def report_goal_state(self, i_new):
        self.found_path = True
        cost = self.tree.cost_to_verts[i_new]
        logger.debug("Found state in goal region with cost %s", cost)

    def get_goal_state_index(self):
        vertices = self.tree.get_vertices()
        mask_vertices_goal = self.goal_region.mask_is_within(vertices)
        if mask_vertices_goal.any():
            indices = mask_vertices_goal.nonzero()[0]
            costs = self.tree.cost_to_verts[indices]
            i = indices[np.argmin(costs)]
        else:
            i = None
        return i

    def get_path(self):
        state_start, goal_region = self.state_start, self.goal_region
        i = self.get_goal_state_index()
        if i is not None:
            path = self.tree.find_path_to_root_from_vertex_index(i)
            path = path[::-1]
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def get_planning_meta_data(self):
        i = self.get_goal_state_index()
        cost = self.tree.cost_to_verts[i] if i is not None else np.inf
        return {"cost_best_path": cost}
