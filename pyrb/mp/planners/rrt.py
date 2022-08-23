import logging

import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus
from pyrb.mp.utils.trees.tree import Tree

RRT_PLANNER_LOGGER_NAME = __file__
logger = logging.getLogger(RRT_PLANNER_LOGGER_NAME)


class LocalPlanner:

    def __init__(self, min_path_distance, min_coll_step_size, max_distance):
        self.min_path_distance = min_path_distance
        self.min_coll_step_size = min_coll_step_size
        self.max_distance = max_distance

    def plan(self, space, state_src, state_dst, max_distance=None):
        # assumes state_src is collision free
        max_distance = max_distance or self.max_distance
        path = self.generate_path(space, state_src, state_dst, max_distance)
        status, validated_path = self.collision_check_path(space, path)
        if status == LocalPlannerStatus.ADVANCED and np.isclose(path[-1], state_dst).all():
            status = LocalPlannerStatus.REACHED
        return status, validated_path

    def collision_check_path(self, space, path):
        status = LocalPlannerStatus.TRAPPED
        cnt = 1
        for state_src, state_dst in zip(path[:-1], path[1:]):
            collision_free_transition = space.is_collision_free_transition(
                state_src=state_src,
                state_dst=state_dst,
                min_step_size=self.min_coll_step_size
            )
            if collision_free_transition:
                status = LocalPlannerStatus.ADVANCED
                cnt += 1
            else:
                status = LocalPlannerStatus.TRAPPED
                break
        return status, path[1:cnt]  # Dont include src

    def generate_path(self, space, state_src, state_dst, max_distance):
        distance_to_dst = space.distance(state_src, state_dst)
        if distance_to_dst < max_distance:
            # reachable
            nr_points = int(max(distance_to_dst / self.min_path_distance, 2))
            path = np.linspace(state_src, state_dst, nr_points)
        else:
            # need to point to
            alpha = max_distance / distance_to_dst
            nr_points = int(max(max_distance / self.min_path_distance, 2))
            state_dst = state_src * (1 - alpha) + state_dst * alpha
            path = np.linspace(state_src, state_dst, nr_points)
        return path


class LocalPlannerSpaceTime(LocalPlanner):

    def __init__(self, max_actuation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_actuation = max_actuation

    def generate_path(self, space, state_src, state_dst, max_distance):
        max_actuation = self.max_actuation
        # TODO: think of a better way to do this...
        t_dst = state_dst[-1]
        if isinstance(max_distance, tuple):
            max_distance, max_time_horizon = max_distance
            t_end = space.transition(t_dst, dt=max_time_horizon)
        else:
            t_end = t_dst
        path = [state_src]
        state_prev = state_src
        config_dst = state_dst[:-1]
        acc_distance = 0
        while True:
            config_prev, t_prev = state_prev[:-1], state_prev[-1]
            if acc_distance < max_distance:
                config_delta = np.clip(config_dst - config_prev, -max_actuation, max_actuation)
                config_nxt = config_prev + config_delta
                acc_distance += np.linalg.norm(config_delta)
            else:
                config_nxt = config_prev
            t_nxt = space.transition(t_prev)
            state_nxt = np.append(config_nxt, t_nxt)
            path.append(state_nxt)
            state_prev = state_nxt
            # is_in_global_goal = state_global_goal is not None and is_vertex_in_goal_region(
            #     config_nxt,
            #     state_global_goal[:-1],
            #     self.global_goal_region_radius
            # ) and t_nxt == state_global_goal[-1]
            if t_nxt == t_end:
                break
        return np.vstack(path)


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

    def run(self):
        state_free = self.space.sample_collision_free_state()
        i_nearest, state_nearest = self.space.find_nearest_state(self.tree.get_vertices(), state_free)
        if i_nearest is None or state_nearest is None:
            return
        status, local_path = self.local_planner.plan(
            self.space,
            state_nearest,
            state_free
            # goal_region,
        )
        if local_path.size > 0:
            state_new = local_path[-1]
            _ = self.tree.append_vertex(state_new, i_parent=i_nearest)
            if self.goal_region.is_within(state_new):
                self.found_path = True
                logger.debug("Found state in goal region!")

    def get_path(self):
        state_start, goal_region = self.state_start, self.goal_region
        vertices = self.tree.get_vertices()
        distances = np.linalg.norm(goal_region.state - vertices, axis=1)
        mask_vertices_goal = distances < goal_region.radius
        if mask_vertices_goal.any():
            i = mask_vertices_goal.nonzero()[0][0]
            path = self.tree.find_path_to_root_from_vertex_index(i)
            path = path[::-1]
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def get_planning_meta_data(self):
        return {
            "nr_verts": self.tree.vert_cnt
        }
