import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus

import logging

LOGGER_NAME_POST_PROCESSING = __file__
logger = logging.getLogger(LOGGER_NAME_POST_PROCESSING)


def space_time_compute_path_cost(path):
    return np.linalg.norm(path[1:, :-1] - path[:-1, :-1], axis=1).sum()


def compute_path_cost(path):
    return np.linalg.norm(path[1:] - path[:-1], axis=1).sum()


def post_process_path_space_time_minimal_time(
        space,
        local_planner,
        goal_region,
        path,
        max_cnt=10,
        max_cnt_no_improvement=0
):
    path_pp = path.copy()
    cnt = 0
    cnt_no_improvement = 0
    # start to naively tie together start and end of path
    i = 0
    j = path_pp.shape[0] - 1
    path_cost = space_time_compute_path_cost(path_pp)
    while cnt < max_cnt or cnt_no_improvement < max_cnt_no_improvement:
        state_src = path_pp[i, :]
        state_dst = path_pp[j, :]
        status, local_path = local_planner.plan(
            space,
            state_src=state_src,
            state_dst=state_dst,
            goal_region=goal_region,
            max_distance=np.inf
        )
        improvement = False
        if status == LocalPlannerStatus.REACHED:
            path_pp_new = np.vstack([path_pp[:i+1], local_path, path_pp[j+1:]])
            path_new_cost = space_time_compute_path_cost(path_pp_new)
            improvement = path_new_cost < path_cost
            if improvement:
                logger.debug("Iter %i Cost improved from %.2f to %.2f", cnt, path_cost, path_new_cost)
                path_cost = path_new_cost
                path_pp = path_pp_new
        elif status == LocalPlannerStatus.ADVANCED and goal_region.is_within(local_path[-1]):
            path_pp_new = np.vstack([path_pp[:i + 1], local_path])
            path_new_cost = space_time_compute_path_cost(path_pp_new)
            improvement = path_new_cost < path_cost
            if improvement:
                logger.debug("Iter %i Cost improved from %.2f to %.2f", cnt, path_cost, path_new_cost)
                path_cost = path_new_cost
                path_pp = path_pp_new
        if not improvement:
            cnt_no_improvement += 1
        else:
            cnt_no_improvement = 0
        if path_pp.shape[0] < 3:
            break
        i = np.random.randint(0, path_pp.shape[0] - 2)
        j = np.random.randint(i + 1, path_pp.shape[0])
        cnt += 1
    logging.debug("Path improvement ratio %0.2f",  path_cost / space_time_compute_path_cost(path))
    return local_planner.interpolate_path_from_waypoints(path_pp)


def post_process_path(
        space,
        local_planner,
        path,
        goal_region,
        max_cnt=10,
        max_cnt_no_improvement=0
):
    path_pp = path.copy()
    cnt = 0
    cnt_no_improvement = 0
    # start to naively tie together start and end of path
    i = 0
    j = path_pp.shape[0] - 1
    path_cost = compute_path_cost(path_pp)
    while cnt < max_cnt or cnt_no_improvement < max_cnt_no_improvement:
        state_src = path_pp[i, :]
        state_dst = path_pp[j, :]
        status, local_path = local_planner.plan(
            space,
            state_src=state_src,
            state_dst=state_dst,
            goal_region=goal_region,
            max_distance=np.inf
        )
        improvement = False
        if status == LocalPlannerStatus.REACHED:
            path_pp_new = np.vstack([path_pp[:i + 1], local_path, path_pp[j + 1:]])
            path_new_cost = compute_path_cost(path_pp_new)
            improvement = path_new_cost < path_cost
            if improvement:
                logger.debug("Iter %i Cost improved from %.2f to %.2f", cnt, path_cost, path_new_cost)
                path_cost = path_new_cost
                path_pp = path_pp_new
        elif status == LocalPlannerStatus.ADVANCED and goal_region.is_within(local_path[-1]):
            path_pp_new = np.vstack([path_pp[:i + 1], local_path])
            path_new_cost = compute_path_cost(path_pp_new)
            improvement = path_new_cost < path_cost
            if improvement:
                logger.debug("Iter %i Cost improved from %.2f to %.2f", cnt, path_cost, path_new_cost)
                path_cost = path_new_cost
                path_pp = path_pp_new
        if not improvement:
            cnt_no_improvement += 1
        else:
            cnt_no_improvement = 0
        if path_pp.shape[0] < 3:
            break
        i = np.random.randint(0, path_pp.shape[0] - 2)
        j = np.random.randint(i + 1, path_pp.shape[0])
        cnt += 1
    logging.debug("Path improvement ratio %0.2f", path_cost / compute_path_cost(path))
    return path_pp


def post_process_path_continuous(
        space,
        local_planner,
        path,
        goal_region,
        max_cnt=10,
        max_cnt_no_improvement=0
):
    path_pp = path.copy()
    cnt = 0
    cnt_no_improvement = 0
    # start to naively tie together start and end of path
    # i = 0
    # j = path_pp.shape[0] - 1
    i_segment, j_segment, state_src, state_dst = 0, path.shape[0] - 2, path[0, :], path[-1, :]
    path_cost = compute_path_cost(path_pp)
    while cnt < max_cnt or cnt_no_improvement < max_cnt_no_improvement:
        status, local_path = local_planner.plan(
            space,
            state_src=state_src,
            state_dst=state_dst,
            goal_region=goal_region,
            max_distance=np.inf
        )
        improvement = False
        if status == LocalPlannerStatus.REACHED:
            path_pp_new_arrs = [
                path_pp[:(i_segment + 1), :]
            ]
            if not np.isclose(path_pp[i_segment], state_src).all():
                path_pp_new_arrs.extend([state_src.reshape(1, -1), state_dst.reshape(1, -1)])
            else:
                path_pp_new_arrs.append(state_dst.reshape(1, -1))
            if not np.isclose(state_dst, path_pp[j_segment + 1]).all():
                path_pp_new_arrs.append(path_pp[j_segment + 1:, :])
            else:
                path_pp_new_arrs.append(path_pp[j_segment + 2:, :])
            path_pp_new = np.vstack(path_pp_new_arrs)
            path_new_cost = compute_path_cost(path_pp_new)
            improvement = path_new_cost < path_cost
            if improvement:
                logger.debug("Iter %i Cost improved from %.2f to %.2f", cnt, path_cost, path_new_cost)
                path_cost = path_new_cost
                path_pp = path_pp_new
        if not improvement:
            cnt_no_improvement += 1
        else:
            cnt_no_improvement = 0
        if path_pp.shape[0] < 3:
            break
        i_segment, j_segment, state_src, state_dst = sample_line_uniformly_on_path(path_pp)
        cnt += 1
    logging.debug("Path improvement ratio %0.2f", path_cost / compute_path_cost(path))
    return path_pp


def sample_line_uniformly_on_path(path):
    path_distances = np.linalg.norm(path[1:] - path[:-1], axis=1)
    path_cumsum = np.cumsum(path_distances)
    total_distance = path_distances.sum()
    distance_x = np.random.uniform(0, path_cumsum[-2])
    i_segment = np.where((path_cumsum >= distance_x))[0][0]
    distance_y = np.random.uniform(path_cumsum[i_segment + 1], total_distance)
    j_segment = i_segment + 1 + np.where((distance_y <= path_cumsum[i_segment + 1:]))[0][0]
    x = interpolate_point_from_distance(distance_x, i_segment, path, path_cumsum)
    y = interpolate_point_from_distance(distance_y, j_segment, path, path_cumsum)
    return i_segment, j_segment, x, y


def interpolate_point_from_distance(cum_distance, segment_nr, path, path_cumsum):
    v1 = path[segment_nr, :]
    v2 = path[segment_nr + 1, :]
    distance_segment = np.linalg.norm(v2 - v1)
    distance = cum_distance - path_cumsum[segment_nr - 1] if segment_nr > 0 else cum_distance
    beta = distance / distance_segment
    if not (0 <= beta <= 1.001):
        assert False, f"{beta}"
    v = (1 - beta) * v1 + beta * v2
    return v


# def post_process_path_space_time_minimal_time(space, local_planner, goal_region, path, max_cnt=10):
#     path_pp = path.copy()
#     cnt = 0
#     cache = []
#     while cnt < max_cnt:
#         i = np.random.randint(0, path_pp.shape[0] - 2)
#         j = np.random.randint(i + 1, path_pp.shape[0])
#         if any(i_old < i and j < j_old for i_old, j_old in cache):
#             print("In cache")
#             cnt += 1
#             continue
#         state_src = path_pp[i, :]
#         state_dst = path_pp[j, :]
#         status, local_path = local_planner.plan(
#             space,
#             state_src=state_src,
#             state_dst=state_dst,
#             goal_region=goal_region,
#             max_distance=np.inf
#         )
#         if status == LocalPlannerStatus.REACHED:
#             k = 0
#             while k < len(cache):
#                 i_old, j_old = cache[k]
#                 no_intersection = j < i_old or i > j_old
#                 if no_intersection:
#                     k += 1
#                 else:
#                     cache.pop(k)
#             path_pp_len_old = path_pp.shape[0]
#             path_pp = np.vstack([path_pp[:i+1], local_path, path_pp[j+1:]])
#             path_pp_len_new = path_pp.shape[0]
#             elements_diff = path_pp_len_old - path_pp_len_new
#             for k in range(len(cache)):
#                 i_old, j_old = cache[k]
#                 if j < i_old:
#                     cache[k] = (i_old - elements_diff, j_old - elements_diff)
#             cache.append((i, j))
#         elif status == LocalPlannerStatus.ADVANCED and goal_region.is_within(local_path[-1]):
#             path_pp = np.vstack([path_pp[:i + 1], local_path, path_pp[j + 1:]])
#         cnt += 1
#     print(cache)
