import logging

import numpy as np

from pyrb.mp.utils.constants import LocalPlannerStatus

LOGGER_NAME_POST_PROCESSING = __file__
logger = logging.getLogger(LOGGER_NAME_POST_PROCESSING)


def space_time_compute_path_cost(path):
    return np.linalg.norm(path[1:, :-1] - path[:-1, :-1], axis=1).sum()


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


def post_process_path_discretized_sampling(
        path,
        space,
        local_planner,
        goal_region,
        max_cnt=10,
        max_cnt_no_improvement=0
    ):
    pass





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