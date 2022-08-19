import math
from enum import Enum, auto

import numpy as np

from pyrb.mp.utils.utils import is_vertex_in_goal_region
from pyrb.mp.planners.static.local_planners import LocalRRTConnectPlannerStatus


class LocalPlanner:

    def __init__(
            self,
            world,
            min_step_size,
            max_distance,
            global_goal_region_radius,
            max_actuation,
            nr_coll_steps=10
    ):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size  # TODO: unused...
        self.world = world
        self.max_actuation = max_actuation
        self.nr_coll_steps = nr_coll_steps

    def plan(self, state_src, state_dst, config_global_goal=None, full_plan=False):
        max_nr_steps = self.compute_max_nr_steps(state_src, state_dst, full_plan)
        state_prev = state_src
        config_dst = state_dst[:-1]
        path = np.zeros((max_nr_steps, state_prev.size))
        cnt = 0
        for delta_t in range(1, max_nr_steps + 1):
            config_prev = state_prev[:-1]
            t_prev = state_prev[-1]
            config_delta = np.clip(config_dst - config_prev, -self.max_actuation, self.max_actuation)
            config_nxt = config_prev + config_delta
            state_nxt = np.append(config_nxt, t_prev + 1)
            collision_free_transition = self.world.is_collision_free_transition(
                state_src=state_prev,
                state_dst=state_nxt,
                nr_coll_steps=self.nr_coll_steps
            )
            is_in_global_goal = config_global_goal is not None and is_vertex_in_goal_region(
                config_nxt,
                config_global_goal,
                self.global_goal_region_radius
            )
            if np.abs(config_delta).sum() == 0:
                # TODO: tmp hack.... NEEDS TO BE FIXED!!!
                break
            if collision_free_transition:
                state_prev = state_nxt
                path[cnt, :] = state_nxt
                cnt += 1
            if is_in_global_goal or not collision_free_transition:
                break
        return path[:cnt, :]

    def compute_max_nr_steps(self, state_src, state_dst, full_plan):
        config_src = state_src[:-1]
        config_dst = state_dst[:-1]
        config_delta = config_dst - config_src
        t_src = state_src[-1]
        t_dst = state_dst[-1]
        nr_steps = t_dst - t_src
        distance = np.linalg.norm(config_delta)
        if not full_plan:
            distance = min(distance, self.max_distance)
        nr_steps_full_act = math.ceil(distance / self.max_actuation)
        max_nr_steps = int(min(nr_steps, nr_steps_full_act))
        return max_nr_steps


class TimeModes(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class LocalPlannerRRTConnect:

    def __init__(
            self,
            world,
            min_step_size,
            max_distance,
            global_goal_region_radius,
            max_actuation,
            nr_coll_steps=10
    ):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size  # TODO: unused...
        self.world = world
        self.max_actuation = max_actuation
        self.nr_coll_steps = nr_coll_steps

    def plan(self, state_src, state_dst, time_mode, state_global_goal=None, full_plan=False):
        state_prev = state_src
        config_dst = state_dst[:-1]

        max_nr_steps = math.ceil(self.max_distance/self.max_actuation) if not full_plan else np.inf
        path = []
        collision_free_transition = True
        has_reached_dst = False
        step_nr = 0
        is_passed_time_horizon = False

        t_dst = state_dst[-1]

        while step_nr < max_nr_steps and collision_free_transition and not has_reached_dst and not is_passed_time_horizon:
            config_prev = state_prev[:-1]
            t_prev = state_prev[-1]
            config_delta = np.clip(config_dst - config_prev, -self.max_actuation, self.max_actuation)
            config_nxt = config_prev + config_delta
            if time_mode == TimeModes.FORWARD:
                t_nxt = t_prev + 1
            else:
                t_nxt = t_prev - 1
            state_nxt = np.append(config_nxt, t_nxt)
            collision_free_transition = self.world.is_collision_free_transition(
                state_src=state_prev,
                state_dst=state_nxt,
                nr_coll_steps=self.nr_coll_steps
            )

            # is_in_global_goal = state_global_goal is not None and is_vertex_in_goal_region(
            #     config_nxt,
            #     state_global_goal[:-1],
            #     self.global_goal_region_radius
            # ) and t_nxt == state_global_goal[-1]

            if collision_free_transition:
                state_prev = state_nxt
                path.append(state_nxt)
            else:
                path.clear()

            step_nr += 1
            has_reached_dst = np.isclose(state_nxt, state_dst).all()

            if time_mode == TimeModes.FORWARD:
                is_passed_time_horizon = t_nxt >= t_dst
            else:
                is_passed_time_horizon = t_nxt <= t_dst

        if collision_free_transition and has_reached_dst:
            status = LocalRRTConnectPlannerStatus.REACHED
        elif collision_free_transition and not has_reached_dst:
            status = LocalRRTConnectPlannerStatus.ADVANCED
        else:
            status = LocalRRTConnectPlannerStatus.TRAPPED
        path = np.vstack(path) if path else np.array([]).reshape((0, state_dst.size))
        return status, path

