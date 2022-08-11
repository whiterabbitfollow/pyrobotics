import bisect
from collections import defaultdict
import itertools
import numpy as np
import math

from typing import Tuple


def heuristic_cost_to_go(q_1, q_2):
    return np.linalg.norm(np.array(q_1) - np.array(q_2))


def extract_path(parents, state_from, state_to):
    state = state_from
    path = [state]
    while state != state_to:
        state = parents[state]
        path.append(state)
    return path[::-1]


def solve(mpenv, s_start: Tuple, s_goal: Tuple):
    q_start, t_start = s_start
    nr_joints = len(q_start)
    actuator_limits = np.hstack([
        mpenv.agent.action_space.low.reshape(-1, 1),
        mpenv.agent.action_space.high.reshape(-1, 1),
        np.zeros((nr_joints,)).reshape(-1, 1)
    ])
    closed = set()
    s_arr = np.hstack(s_start)
    s_goal_arr = np.hstack(s_goal)
    all_actions = list(itertools.product(actuator_limits[0, :], actuator_limits[1, :]))
    all_actions.remove((0,) * nr_joints)
    cost_left = heuristic_cost_to_go(s_goal_arr, s_arr)
    queue = [(cost_left, (q_start, t_start))]
    g = defaultdict(lambda: math.inf)
    g[(q_start, t_start)] = 0
    parents = {}
    while queue:
        cost, state = queue.pop(0)
        (q_curr, t_curr) = state
        s_arr_curr = np.hstack(state)
        q_arr_curr = np.array(q_curr)
        if heuristic_cost_to_go(s_goal_arr, s_arr_curr) < 0.1:
            return extract_path(parents, (q_curr, t_curr), (q_start, t_start))
        closed.add((q_curr, t_curr))
        t_nxt = t_curr + 1
        for action in all_actions:
            q_arr_nxt = q_arr_curr + action
            q_nxt = tuple(q_arr_nxt)
            s_nxt = (q_nxt, t_nxt)
            s_arr_nxt = np.hstack(s_nxt)
            if not mpenv.is_state_valid(q_nxt):
                continue
            if (q_nxt, t_nxt) in closed:
                continue
            is_free = mpenv.is_edge_free(t_curr, q_arr_curr, q_arr_nxt)
            if not is_free:
                continue
            actuator_cost = np.linalg.norm(action)
            time_step_cost = 0.1
            transition_cost = actuator_cost + time_step_cost
            cost = g[(q_curr, t_curr)] + transition_cost
            if cost < g[(q_nxt, t_nxt)]:
                g[(q_nxt, t_nxt)] = cost
                parents[(q_nxt, t_nxt)] = (q_curr, t_curr)
                cost_left = heuristic_cost_to_go(s_goal_arr, s_arr_nxt)
                bisect.insort_left(queue, (cost + cost_left, s_nxt))
    return None


def solve_with_distance(mpenv, q_start: Tuple, q_goal: Tuple, t_start=0):
    nr_joints = len(q_start)
    actuator_limits = np.hstack([
        mpenv.agent.action_space.low.reshape(-1, 1),
        mpenv.agent.action_space.high.reshape(-1, 1),
        np.zeros((nr_joints,)).reshape(-1, 1)
    ])
    space_limits = mpenv.agent.joint_limits
    closed = set()
    q_arr = np.array(q_start)
    q_goal_arr = np.array(q_goal)
    all_actions = list(itertools.product(actuator_limits[0, :], actuator_limits[1, :]))
    all_actions.remove((0,) * nr_joints)
    cost_left = heuristic_cost_to_go(q_goal_arr, q_arr)
    queue = [(cost_left, (q_start, t_start))]
    g = defaultdict(lambda: math.inf)
    g[(q_start, t_start)] = 0
    parents = {}
    while queue:
        cost, (q_curr, t_curr) = queue.pop(0)
        q_arr_curr = np.array(q_curr)
        if heuristic_cost_to_go(q_goal_arr, q_arr_curr) < 0.1:
            return extract_path(parents, (q_curr, t_curr), (q_start, t_start))
        closed.add((q_curr, t_curr))
        t_nxt = t_curr + 1
        for action in all_actions:
            q_arr_nxt = q_arr_curr + action
            q_nxt = tuple(q_arr_nxt)
            if (q_nxt < space_limits[:, 0]).any() or (q_nxt > space_limits[:, 1]).any():
                continue
            if (q_nxt, t_nxt) in closed:
                continue
            is_collision, _, distance = mpenv.collision_check_transition_with_distance(q_arr_curr, q_arr_nxt)
            if is_collision:
                continue
            actuator_cost = np.linalg.norm(action)
            transition_cost = actuator_cost
            distance_cost = max(distance, 1e-6)
            distance_cost = (0.01 / distance_cost) * 0.1
            transition_cost += distance_cost
            cost = g[(q_curr, t_curr)] + transition_cost
            if cost < g[(q_nxt, t_nxt)]:
                g[(q_nxt, t_nxt)] = cost
                parents[(q_nxt, t_nxt)] = (q_curr, t_curr)
                cost_left = heuristic_cost_to_go(q_goal_arr, q_arr_nxt)
                bisect.insort_left(queue, (cost + cost_left, (q_nxt, t_nxt)))
    return None


def solve_static(mpenv, q_start: Tuple, q_goal: Tuple):
    nr_joints = len(q_start)
    actuator_limits = np.hstack([
        mpenv.agent.action_space.low.reshape(-1, 1),
        mpenv.agent.action_space.high.reshape(-1, 1),
        np.zeros((nr_joints,)).reshape(-1, 1)
    ])
    space_limits = mpenv.agent.joint_limits
    closed = set()
    q_arr = np.array(q_start)
    q_goal_arr = np.array(q_goal)
    all_actions = list(itertools.product(actuator_limits[0, :], actuator_limits[1, :]))
    all_actions.remove((0,) * nr_joints)
    cost_left = heuristic_cost_to_go(q_goal_arr, q_arr)
    queue = [(cost_left, q_start)]
    g = defaultdict(lambda: math.inf)
    g[q_start] = 0
    parents = {}
    while queue:
        cost, q_curr = queue.pop(0)
        q_arr_curr = np.array(q_curr)
        if heuristic_cost_to_go(q_goal_arr, q_arr_curr) < 0.1:
            return extract_path(parents, q_curr, q_start)
        closed.add(q_curr)
        for action in all_actions:
            q_arr_nxt = q_arr_curr + action
            q_nxt = tuple(q_arr_nxt)
            if (q_nxt < space_limits[:, 0]).any() or (q_nxt > space_limits[:, 1]).any():
                continue
            if q_nxt in closed:
                continue
            is_collision, _, distance = mpenv.collision_check_transition_with_distance(q_arr_curr, q_arr_nxt)
            if is_collision:
                continue

            actuator_cost = np.linalg.norm(action)
            transition_cost = actuator_cost

            distance_cost = max(distance, 1e-6)
            distance_cost = (0.01 / distance_cost) * 0.1
            transition_cost += distance_cost

            cost = g[q_curr] + transition_cost
            if cost < g[q_nxt]:
                g[q_nxt] = cost
                parents[q_nxt] = q_curr
                cost_left = heuristic_cost_to_go(q_goal_arr, q_arr_nxt)
                bisect.insort_left(queue, (cost + cost_left, q_nxt))
    return None


if __name__ == "__main__":

    from dev.mp_time_envs import RobAndRobCollabMPEnv
    from envs import make_2d_env_dynamic_grid
    env = make_2d_env_dynamic_grid(nr_lagging_steps=1, grid_size=64)
    mpenv = RobAndRobCollabMPEnv(agent=env.agent, adversary=env.adversary)
    env.reset()
    t_start = 0
    mpenv.set_time(t_start)
    config_start = mpenv.sample_collision_free_config()
    s_start = (tuple(config_start), t_start)

    t_goal = 300
    mpenv.set_time(t_goal)
    config_goal = mpenv.sample_collision_free_config()
    s_goal = (tuple(config_goal), t_goal)
    print(config_start, config_goal)
    import time
    time_s = time.time()
    path = solve(mpenv, s_start, s_goal)
    print(f"tims elapsed {time.time()-time_s}")
    print(path)
    env.agent.set_config(config_start)
    env.set_goal_state(config_goal)
    env.adversary.set_time(0)
    acc_reward = 0
    print(path[0])
    for (q, _), (q_nxt, _) in zip(path[:-1], path[1:]):
        action = np.array(q_nxt) - np.array(q)
        obs, reward, done, info = env.step(action)
        acc_reward += reward
        env.render(mode="human", matplotlib_pause=0.1)
    print(acc_reward)

    # print(env.reset())
    # print(env.observation_space["observation"])
    # visualize_collision_free_space_static(env)
    # [ 3.07670291 -1.09267902]
    # conf = np.array([2.98925226, 1.3335978])
    # env.adversary.set_config(conf)
    # start_state = env.get_config()
    # cylinder_radii = 0.0001
    # mesh = trimesh.creation.cylinder(cylinder_radii, height=0.1)
    # print(env.adversary.collision_manager.in_collision_single(mesh))
    # print(env.adversary.collision_manager.min_distance_single(mesh))

    # path = solve_static(
    #         env, tuple(start_state), tuple(env.goal_state)
    #     )
    # assert path is not None
    # path = np.array(
    #     path
    # )
    # env.set_config(start_state)
    # from envs.utils import vizualize_path
    # vizualize_path(env, path, matplotlib_pause=0.1)
    # assert path is not None
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # path = np.array(path)
    # plt.plot(path[:, 0], path[:, 1])
    # plt.show()
    #
    #
