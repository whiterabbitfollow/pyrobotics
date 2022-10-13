import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus


class LocalPlanner:

    def __init__(self, min_coll_step_size, max_distance):
        self.min_coll_step_size = min_coll_step_size
        self.max_distance = max_distance

    def plan(self, space, state_src, state_dst, max_distance=None, goal_region=None):
        # assumes state_src is collision free
        max_distance = max_distance or self.max_distance
        distance_to_dst = space.distance(state_src, state_dst)
        state_dst_target = state_dst
        if distance_to_dst > max_distance:
            alpha = max_distance / distance_to_dst
            state_dst_target = state_src * (1 - alpha) + state_dst * alpha
        collision_free_transition = space.is_collision_free_transition(
            state_src=state_src,
            state_dst=state_dst_target,
            min_step_size=self.min_coll_step_size
        )
        path = state_dst_target.reshape(1, -1)
        status = self.coll_check_status(collision_free_transition, path, state_dst)
        return status, path

    def coll_check_status(self, collision_free_transition, path, state_dst):
        state_final_path = path[-1]
        if collision_free_transition and np.isclose(state_final_path, state_dst).all():
            status = LocalPlannerStatus.REACHED
        elif collision_free_transition:
            status = LocalPlannerStatus.ADVANCED
        else:
            status = LocalPlannerStatus.TRAPPED
        return status


class LocalPlannerSpaceTime(LocalPlanner):

    def __init__(self, max_actuation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_actuation = max_actuation

    def plan(self, space, state_src, state_dst, max_distance=None, goal_region=None):
        # assumes state_src is collision free
        status, waypoints = self.generate_waypoints(
            space, state_src, state_dst, max_distance, goal_region
        )
        return status, waypoints[1:]

    def generate_waypoints(self, space, state_src, state_dst, max_distance=None, goal_region=None):
        t_src, t_dst = state_src[-1], state_dst[-1]
        if t_src > t_dst:
            # path planning backwards in time
            collision_free, waypoints = self.straight_line_to_destination(
                space,
                state_dst,
                state_src,
                max_distance,
                goal_region
            )
            status = self.coll_check_status(collision_free, waypoints, state_src)
            waypoints = waypoints[::-1]
        else:
            collision_free, waypoints = self.straight_line_to_destination(
                space,
                state_src,
                state_dst,
                max_distance,
                goal_region
            )
            status = self.coll_check_status(collision_free, waypoints, state_dst)
        return status, waypoints

    def straight_line_to_destination(self, space, state_src, state_dst, max_distance=None, goal_region=None):
        t_src, t_dst = state_src[-1], state_dst[-1]
        assert t_src < t_dst
        if max_distance is None:
            max_distance, max_time_horizon = self.max_distance
            t_end = self.transition(t_src, dt=max_time_horizon, time_horizon=min(space.max_time, t_dst))
        else:
            t_end = t_dst
        waypoints = [state_src]
        state_prev = state_src
        config_dst = state_dst[:-1]
        acc_distance = 0
        # TODO: add waypoints...
        while True:
            config_prev, t_prev = state_prev[:-1], state_prev[-1]
            config_nxt = self.calculate_next_configuration(config_prev, config_dst)
            config_delta = config_nxt - config_prev
            config_delta_distance = np.linalg.norm(config_delta)
            acc_distance += np.linalg.norm(config_delta_distance)
            t_nxt = self.transition(t_prev, time_horizon=space.max_time)
            state_nxt = np.append(config_nxt, t_nxt)
            is_in_global_goal = goal_region is not None and goal_region.is_within(state_nxt)
            collision_free_transition = space.is_collision_free_transition(
                state_src=state_prev,
                state_dst=state_nxt,
                min_step_size=self.min_coll_step_size
            )
            end = not collision_free_transition or (t_nxt == t_end) or is_in_global_goal
            end = end or acc_distance >= max_distance
            state_prev = state_nxt
            if end:
                waypoints.append(state_nxt)
                break
        return collision_free_transition, np.vstack(waypoints)

    def calculate_next_configuration(self, config_src, config_dst):
        direction = config_dst - config_src
        distance = np.linalg.norm(direction)
        if distance > 0:
            beta = np.clip(self.max_actuation / distance, 0, 1)
        else:
            beta = 0
        return config_src * (1 - beta) + config_dst * beta

    def interpolate_path_from_waypoints(self, path_sparse):
        path = []
        for state_src, state_dst in zip(path_sparse[:-1], path_sparse[1:]):
            t = int(state_dst[-1] - state_src[-1])
            if t == 1:
                path.append(state_src.copy())
            else:
                state = state_src
                config_dst = state_dst[:-1]
                t_dst = state_dst[-1]
                t_src = state[-1]
                cnt = 0
                while state[-1] != t_dst:
                    path.append(state.copy())
                    config = state[:-1]
                    config_nxt = self.calculate_next_configuration(config, config_dst)
                    cnt += 1
                    state = np.append(config_nxt, t_src + cnt)
        path.append(path_sparse[-1])
        return np.vstack(path)

    # def bang_bang_control_to_destination(self, space, state_src, state_dst, max_distance=None, goal_region=None):
    #     t_src, t_dst = state_src[-1], state_dst[-1]
    #     assert t_src < t_dst
    #     if max_distance is None:
    #         max_distance, max_time_horizon = self.max_distance
    #         t_end = self.transition(t_src, dt=max_time_horizon, time_horizon=min(space.max_time, t_dst))
    #     else:
    #         t_end = t_dst
    #     waypoints = [state_src]
    #     state_prev = state_src
    #     config_dst = state_dst[:-1]
    #     acc_distance = 0
    #     config_delta_prev = None
    #     max_actuation = self.max_actuation
    #     while True:
    #         config_prev, t_prev = state_prev[:-1], state_prev[-1]
    #         if acc_distance < max_distance:
    #             config_delta = np.clip(config_dst - config_prev, -max_actuation, max_actuation)
    #             config_nxt = config_prev + config_delta
    #             acc_distance += np.linalg.norm(config_delta)
    #         else:
    #             config_nxt = config_prev
    #             config_delta = config_delta_prev
    #
    #         t_nxt = self.transition(t_prev, time_horizon=space.max_time)
    #         state_nxt = np.append(config_nxt, t_nxt)
    #         is_in_global_goal = goal_region is not None and goal_region.is_within(state_nxt)
    #         collision_free_transition = space.is_collision_free_transition(
    #             state_src=state_prev,
    #             state_dst=state_nxt,
    #             min_step_size=self.min_coll_step_size
    #         )
    #         end = not collision_free_transition or (t_nxt == t_end) or is_in_global_goal
    #         changed_traj = config_delta_prev is not None and not np.isclose(config_delta_prev, config_delta).all()
    #         if changed_traj or end:
    #             if end:
    #                 waypoints.append(state_nxt)
    #             else:
    #                 waypoints.append(state_prev)
    #         state_prev = state_nxt
    #         config_delta_prev = config_delta
    #         if end:
    #             break
    #     return collision_free_transition, np.vstack(waypoints)
    #
    # def interpolate_path_from_waypoints(self, path_sparse):
    #     path = []
    #     max_actuation = self.max_actuation
    #     for state_src, state_dst in zip(path_sparse[:-1], path_sparse[1:]):
    #         t = int(state_dst[-1] - state_src[-1])
    #         if t == 1:
    #             path.append(state_src.copy())
    #         else:
    #             state = state_src
    #             t_dst = state_dst[-1]
    #             t_src = state[-1]
    #             cnt = 0
    #             while state[-1] != t_dst:
    #                 path.append(state.copy())
    #                 config_nxt = state[:-1] + np.clip(state_dst[:-1] - state[:-1], -max_actuation, max_actuation)
    #                 cnt += 1
    #                 state = np.append(config_nxt, t_src + cnt)
    #     path.append(path_sparse[-1])
    #     return np.vstack(path)

    def transition(self, t, dt=1, time_horizon=np.inf):
        return min(t + dt, time_horizon)


class LocalPlannerSpaceTimeInftyNorm(LocalPlannerSpaceTime):

    # TODO: only supports 2 dim space so far..
    def calculate_next_configuration(self, config_src, config_dst):
        max_dist = np.linalg.norm(config_dst - config_src)
        if np.isclose(max_dist, 0):
            beta = 0
        else:
            q_d = (config_dst - config_src).ravel()
            q_d = np.abs(q_d)
            w = q_d[0]
            h = q_d[1]
            angle = np.arctan2(h, w)
            if w > h:
                d = self.max_actuation / np.cos(angle)
            else:
                d = self.max_actuation / np.sin(angle)
            beta = d / max_dist
            beta = np.clip(beta, 0, 1)
        q = beta * config_dst + (1 - beta) * config_src
        action = q - config_src
        assert np.round(np.linalg.norm(action, ord=np.inf), 2) <= self.max_actuation
        return q