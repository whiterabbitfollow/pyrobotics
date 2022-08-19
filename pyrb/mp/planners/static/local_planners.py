from enum import Enum, auto

import numpy as np

from pyrb.mp.utils.utils import is_vertex_in_goal_region


class LocalRRTConnectPlannerStatus(Enum):
    TRAPPED = auto()
    ADVANCED = auto()
    REACHED = auto()



class LocalPlanner:

    def __init__(self, world, min_step_size, max_distance, global_goal_region_radius):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size
        self.world = world

    def plan(self, state_src, state_dst, state_global_goal=None, full_plan=False):
        # assumes state_src is collision free
        state_delta = state_dst - state_src
        distance = np.linalg.norm(state_delta)
        if not full_plan:
            distance = min(distance, self.max_distance)
        nr_steps = int(distance / self.min_step_size)
        path, collision_free_transition = np.array([]), False
        state_closest = None
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            collision_free_transition = self.world.is_collision_free_state(state)
            is_in_global_goal = state_global_goal is not None and is_vertex_in_goal_region(
                    state,
                    state_global_goal,
                    self.global_goal_region_radius
                )
            if collision_free_transition:
                state_closest = state
            if is_in_global_goal or not collision_free_transition:
                break
        if state_closest is not None:
            min_transition_distance = 0.2   # TODO: make configurable...
            distance = np.linalg.norm(state_closest - state_src)
            nr_steps = int(distance/min_transition_distance)
            if nr_steps > 1:
                path = np.linspace(state_src, state_closest, nr_steps)
            else:
                path = np.vstack([state_src, state_closest])
            path = path[1:, :]  # Don't include first...
        return path


class LocalPlannerRRTConnect:

    def __init__(self, world, min_step_size, max_distance, global_goal_region_radius):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size
        self.world = world

    def plan(self, state_src, state_dst, state_global_goal=None, full_plan=False):
        # assumes state_src is collision free
        state_delta = state_dst - state_src
        distance = np.linalg.norm(state_delta)
        if not full_plan:
            distance = min(distance, self.max_distance)
        nr_steps = int(distance / self.min_step_size)
        state_closest = None
        status = LocalPlannerStatus.TRAPPED
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            collision_free_transition = self.world.is_collision_free_state(state)
            is_in_global_goal = state_global_goal is not None and is_vertex_in_goal_region(
                state,
                state_global_goal,
                self.global_goal_region_radius
            )
            if collision_free_transition:
                state_closest = state
                status = LocalPlannerStatus.ADVANCED
            else:
                status = LocalPlannerStatus.TRAPPED
                state_closest = None
            if is_in_global_goal or not collision_free_transition:
                break
        if state_closest is not None and np.isclose(state_closest, state_dst).all():
            status = LocalPlannerStatus.REACHED
        return status, state_closest
