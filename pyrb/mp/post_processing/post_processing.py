import logging

import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus


LOGGER_NAME_POST_PROCESSING = __file__
logger = logging.getLogger(LOGGER_NAME_POST_PROCESSING)


# TODO: missing systematic post processing


class PathPostProcessor:

    def __init__(self, space, local_planner, goal_region):
        self.space = space
        self.local_planner = local_planner
        self.goal_region = goal_region
        self.path = None
        self.path_cost_start = np.inf

    def post_process(self, path, max_cnt=10, max_cnt_no_improvement=0):
        self.cnt = 0
        self.cnt_no_improvement = 0
        self.reset()
        self.set_path(path)
        logger.debug("Started post processing")
        while (self.cnt < max_cnt or self.cnt_no_improvement < max_cnt_no_improvement) and self.path.shape[0] > 2:
            indx_segment_src, state_src, indx_segment_dst, state_dst = self.get_points_from_path()
            status, local_path = self.local_planner.plan(
                self.space,
                state_src=state_src,
                state_dst=state_dst,
                # goal_region=goal_region,
                max_distance=np.inf
            )
            if status == LocalPlannerStatus.REACHED:
                path = self.merge_segment_into_path(path, indx_segment_src, state_src, indx_segment_dst, state_dst)
                self.set_path(path)
                self.cnt_no_improvement = 0
            else:
                self.cnt_no_improvement += 1
            self.cnt += 1
        return self.path.copy()

    def reset(self):
        self.path = None
        self.path_cost_start = np.inf

    def set_path(self, path):
        if self.path is None:
            self.path_cost_start = self.compute_path_cost(path)
            logger.debug("Setting path")
        else:
            logger.debug("Path ratio %0.2f", self.compute_path_cost(self.path) / self.path_cost_start)
        self.path = path

    def compute_path_cost(self, path):
        return np.linalg.norm(path[1:] - path[:-1], axis=1).sum()

    def get_points_from_path(self):
        path = self.path
        if self.cnt == 0:
            # start to naively tie together start and end of path
            indx_segment_src = 0
            indx_segment_dst = path.shape[0] - 1
        else:
            indx_segment_src = np.random.randint(0, path.shape[0] - 2)
            indx_segment_dst = np.random.randint(indx_segment_src + 1, path.shape[0])
        state_src = path[indx_segment_src, :]
        state_dst = path[indx_segment_dst, :]
        return indx_segment_src, state_src, indx_segment_dst, state_dst

    def merge_segment_into_path(self, path, indx_segment_src, state_src, indx_segment_dst, state_dst):
        path_pp_new_arrs = [
            path[:(indx_segment_src + 1), :]
        ]
        if np.isclose(path[indx_segment_src], state_src).all():
            path_pp_new_arrs.append(state_dst.reshape(1, -1))
        else:
            path_pp_new_arrs.extend([state_src.reshape(1, -1), state_dst.reshape(1, -1)])
        if np.isclose(state_dst, path[indx_segment_dst + 1]).all():
            path_pp_new_arrs.append(path[indx_segment_dst + 2:, :])
        else:
            path_pp_new_arrs.append(path[indx_segment_dst + 1:, :])
        return np.vstack(path_pp_new_arrs)



class PathPostProcessorRandom(PathPostProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_cumsum = None

    def set_path(self, path):
        super().set_path(path)
        path_distances = np.linalg.norm(path[1:] - path[:-1], axis=1)
        self.path_cumsum = np.cumsum(path_distances)

    def get_points_from_path(self):
        if self.cnt == 0:
            # start to naively tie together start and end of path
            indx_segment_src = 0
            indx_segment_dst = self.path.shape[0] - 1
            state_src = self.path[indx_segment_src, :]
            state_dst = self.path[indx_segment_dst, :]
        else:
            indx_segment_src, state_src, indx_segment_dst, state_dst = self.sample_uniformly_on_path()
        return indx_segment_src, state_src, indx_segment_dst, state_dst

    def sample_uniformly_on_path(self):
        path_cumsum = self.path_cumsum
        distance_x = np.random.uniform(0, path_cumsum[-2])
        i_segment = np.where((distance_x <= path_cumsum))[0][0]
        distance_y = np.random.uniform(path_cumsum[i_segment + 1], path_cumsum[-1])
        j_segment = i_segment + 1 + np.where((distance_y <= path_cumsum[i_segment + 1:]))[0][0]
        x = self.interpolate_point_from_distance(distance_x, i_segment)
        y = self.interpolate_point_from_distance(distance_y, j_segment)
        return i_segment, x, j_segment, y

    def interpolate_point_from_distance(self, cum_distance, segment_nr):
        path_cumsum = self.path_cumsum
        v1 = self.path[segment_nr, :]
        v2 = self.path[segment_nr + 1, :]
        distance_segment = np.linalg.norm(v2 - v1)
        distance = cum_distance - path_cumsum[segment_nr - 1] if segment_nr > 0 else cum_distance
        beta = distance / distance_segment
        if not (0 <= beta <= 1.001):
            assert False, f"{beta}"
        v = (1 - beta) * v1 + beta * v2
        return v


class PathPostProcessorRandomST(PathPostProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_cumsum = None
        self.t_start = None

    def post_process(self, path, max_cnt=10, max_cnt_no_improvement=0):
        self.cnt = 0
        self.cnt_no_improvement = 0
        self.reset()
        self.set_path(path)
        while (self.cnt < max_cnt or self.cnt_no_improvement < max_cnt_no_improvement) and self.path.shape[0] > 2:
            indx_segment_src, state_src, indx_segment_dst, state_dst = self.get_points_from_path()
            # try to connect src with goal
            dt = np.linalg.norm(self.goal_region.state[:-1] - state_src[:-1]) / self.local_planner.max_actuation
            state_dst_goal = np.append(self.goal_region.state[:-1], np.ceil(dt) + state_src[-1])
            # self.goal_region
            status, local_path = self.local_planner.plan(
                self.space,
                state_src=state_src,
                state_dst=state_dst_goal,
                # goal_region=goal_region,
                max_distance=np.inf
            )
            if status == LocalPlannerStatus.REACHED:
                path = self.set_new_tail(path, indx_segment_src, state_src, state_dst_goal)
                self.set_path(path)
                self.cnt_no_improvement = 0
                continue

            status, local_path = self.local_planner.plan(
                self.space,
                state_src=state_src,
                state_dst=state_dst,
                # goal_region=goal_region,
                max_distance=np.inf
            )
            if status == LocalPlannerStatus.REACHED:
                path = self.merge_segment_into_path(path, indx_segment_src, state_src, indx_segment_dst, state_dst)
                self.set_path(path)
                self.cnt_no_improvement = 0
            else:
                self.cnt_no_improvement += 1
            # dt = np.linalg.norm(self.goal_region.q - state_dst[:-1]) / self.local_planner.max_actuation
            # state_dst_goal = np.append(self.goal_region.q, int(dt) + state_dst[-1])
            # status, local_path = self.local_planner.plan(
            #     self.space,
            #     state_src=state_dst,
            #     state_dst=state_dst_goal,
            #     # goal_region=goal_region,
            #     max_distance=np.inf
            # )
            # if status == LocalPlannerStatus.REACHED:
            #     path = self.set_new_tail(path, indx_segment_dst, state_dst, state_dst_goal)
            #     self.set_path(path)
            #     self.cnt_no_improvement = 0
            self.cnt += 1
        return self.path.copy()

    def set_new_tail(self, path, indx_segment_src, state_src, state_tail):
        path_pp_new_arrs = [
            path[:(indx_segment_src + 1), :]
        ]
        if np.isclose(path[indx_segment_src], state_src).all():
            path_pp_new_arrs.append(state_tail.reshape(1, -1))
        else:
            path_pp_new_arrs.extend([state_src.reshape(1, -1), state_tail.reshape(1, -1)])
        return np.vstack(path_pp_new_arrs)

    def compute_path_cost(self, path):
        return np.linalg.norm(path[1:, :-1] - path[:-1, :-1], axis=1).sum()

    def set_path(self, path):
        super().set_path(path)
        ts = path[:, -1]
        t_segments = ts[1:] - ts[:-1]
        self.t_cumsum = np.cumsum(t_segments)
        self.t_start = path[0, -1]

    def get_points_from_path(self):
        dt_src = np.random.randint(0, self.t_cumsum[-2])
        indx_segment_src = np.where(dt_src <= self.t_cumsum)[0][0]
        dt_dst = np.random.randint(self.t_cumsum[indx_segment_src + 1], self.t_cumsum[-1] + 1)
        indx_segment_dst = indx_segment_src + 1 + np.where(dt_dst <= self.t_cumsum[indx_segment_src + 1:])[0][0]
        state_src = self.get_state_from_segment(dt_src, self.path[[indx_segment_src, indx_segment_src + 1], :])
        state_dst = self.get_state_from_segment(dt_dst, self.path[[indx_segment_dst, indx_segment_dst + 1], :])
        return indx_segment_src, state_src, indx_segment_dst, state_dst

    def get_state_from_segment(self, dt, path_segment_sparse):
        path_segment_dense = self.local_planner.interpolate_path_from_waypoints(path_sparse=path_segment_sparse)
        t_segment_start = path_segment_dense[0, -1]
        t_src_segment = (self.t_start + dt - t_segment_start).astype(int)
        state = path_segment_dense[t_src_segment]
        return state

