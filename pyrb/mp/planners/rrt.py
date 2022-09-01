import logging

import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus

RRT_PLANNER_LOGGER_NAME = __file__
logger = logging.getLogger(RRT_PLANNER_LOGGER_NAME)


class LocalPlanner:

    def __init__(self, min_path_distance, min_coll_step_size, max_distance):
        self.min_path_distance = min_path_distance
        self.min_coll_step_size = min_coll_step_size
        self.max_distance = max_distance

    def plan(self, space, state_src, state_dst, max_distance=None, goal_region=None):
        # assumes state_src is collision free
        max_distance = max_distance or self.max_distance
        coll_path = self.generate_path_for_collision_checking(
            space, state_src, state_dst, max_distance, goal_region
        )
        status, validated_path = self.collision_check_path(space, coll_path)
        if validated_path.size:
            path = self.interpolate_path(validated_path)
        else:
            path = validated_path
        if status == LocalPlannerStatus.ADVANCED and np.isclose(path[-1], state_dst).all():
            status = LocalPlannerStatus.REACHED
        return status, path

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

    def generate_path_for_collision_checking(self, space, state_src, state_dst, max_distance, goal_region=None):
        distance_to_dst = space.distance(state_src, state_dst)
        if distance_to_dst < max_distance:
            # reachable
            nr_points = int(max(distance_to_dst / self.min_path_distance + 1, 2))
            path = np.linspace(state_src, state_dst, nr_points)
        else:
            # need to point to
            alpha = max_distance / distance_to_dst
            nr_points = int(max(max_distance / self.min_path_distance + 1, 2))
            state_dst = state_src * (1 - alpha) + state_dst * alpha
            path = np.linspace(state_src, state_dst, nr_points)
        return path

    def interpolate_path(self, validate_path):
        return validate_path


class LocalPlannerSpaceTime(LocalPlanner):

    def __init__(self, max_actuation, *args, nr_time_steps=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_actuation = max_actuation
        self.nr_time_steps = nr_time_steps

    def generate_path_for_collision_checking(self, space, state_src, state_dst, max_distance, goal_region=None):
        t_src, t_dst = state_src[-1], state_dst[-1]
        if t_src > t_dst:
            # path planning backwards in time
            path = self.bang_bang_control_to_destination(
                space,
                state_dst,
                state_src,
                max_distance,
                goal_region
            )
            path = path[::-1]
        else:
            path = self.bang_bang_control_to_destination(
                space,
                state_src,
                state_dst,
                max_distance,
                goal_region
            )
        return path

    def bang_bang_control_to_destination(self, space, state_src, state_dst, max_distance, goal_region=None):
        # TODO: think of a better way to do this...
        max_actuation = self.max_actuation
        t_src, t_dst = state_src[-1], state_dst[-1]
        if isinstance(max_distance, tuple):
            max_distance, max_time_horizon = max_distance
            t_end = self.transition(t_src, dt=max_time_horizon, time_horizon=space.max_time)
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
            t_nxt = self.transition(t_prev, time_horizon=space.max_time)
            state_nxt = np.append(config_nxt, t_nxt)
            path.append(state_nxt)
            state_prev = state_nxt
            is_in_global_goal = goal_region is not None and goal_region.is_within(state_nxt)
            if t_nxt == t_end or is_in_global_goal:
                break
        return np.vstack(path)

    def transition(self, t, dt=1, time_horizon=np.inf):
        return min(t + dt, time_horizon)

    def interpolate_path(self, validate_path):
        state_end = validate_path[-1, :]
        new_path = validate_path[::self.nr_time_steps]
        if (new_path[-1] != state_end).any():
            new_path = np.vstack([new_path, state_end])
        return new_path


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
            state_free,
            goal_region=self.goal_region
        )
        if local_path.size > 0:
            i_parent = i_nearest
            for state_new in local_path:
                edge_cost = self.space.transition_cost(state_new, self.tree.vertices[i_parent])
                if self.can_run() and not self.tree.vertex_exists(state_new):
                    i_parent = self.tree.append_vertex(
                        state_new, i_parent=i_parent, edge_cost=edge_cost
                    )
                    if self.goal_region.is_within(state_new):
                        self.found_path = True
                        logger.debug("Found state in goal region!")

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
