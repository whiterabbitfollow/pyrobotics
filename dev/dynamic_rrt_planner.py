from collections import defaultdict
import numpy as np

import logging

from envs.env3d import Matplotlib3D
from envs.misc import EnvRanges3D

logger = logging.getLogger(__name__)


class DynamicRRTPlanner:

    def __init__(self, env, dt_max=100):
        self.env = env
        self.edges = defaultdict(lambda: [])
        self.parents = dict()
        self.time_window = dt_max
        self.conf_dim = self.env.agent.nr_joints
        self.state_dim = self.conf_dim + 1
        self.max_nr_vertices = int(1e4)
        self.verts_cnt = 0
        self.verts = np.zeros((self.max_nr_vertices, self.state_dim))
        self.max_actuation = self.env.agent.max_actuation
        self.smallest_distance = np.inf
        self.q_goal = None

    def plan(self, s_src, q_goal):
        # Need some criteria to stop...
        self.edges.clear()
        self.parents.clear()
        self.verts[:] = 0
        self.verts[0, :] = s_src
        self.verts_cnt = 1
        self.smallest_distance = np.inf
        self.q_goal = q_goal
        path_linear = self.plan_linearly_to_goal(s_src)
        if path_linear is not None:
            logger.debug("Linear path to goal possible")
            return path_linear
        else:
            logger.debug("Couldn't go linearly to goal")
        while True:
            s_sample = self.sample_collision_free_state()
            s_nearest = self.get_nearest_vertex(s_sample)
            if s_nearest is None:
                continue
            states_list = self.plan_motion(s_nearest, s_sample)
            if states_list:
                states = np.vstack(states_list)
                self.add_states_to_tree(s_nearest, states)
                s_goal = self.check_states_if_in_goal_set(states, q_goal)
                if s_goal is not None:
                    path = self.get_path(s_goal, s_src)
                    return path

    def plan_linearly_to_goal(self, s_src):
        s_goal = np.hstack((self.q_goal, self.time_window))
        states_list = self.plan_motion(s_src, s_goal)
        if states_list:
            states = np.vstack(states_list)
            self.add_states_to_tree(s_src, states)
            s_goal = self.check_states_if_in_goal_set(states, q_goal)
            if s_goal is not None:
                path = self.get_path(s_goal, s_src)
                return path

    def sample_collision_free_state(self):
        t = np.random.randint(0, self.time_window)
        self.env.set_time(t)
        s = np.zeros((self.state_dim,))
        s[:self.conf_dim] = self.env.sample_collision_free_config()
        s[-1] = t
        return s

    def get_nearest_vertex(self, s_sample):
        verts = self.verts[:self.verts_cnt]
        dt = s_sample[-1] - verts[:, -1]
        valid_states = dt > 0
        if valid_states.sum() == 0:
            return None
        # what if none is valid?
        dist = np.linalg.norm(verts[valid_states, :self.conf_dim] - s_sample[:self.conf_dim], axis=1)
        indx_nearest = np.argmin(dist)
        s_nearest = verts[valid_states][indx_nearest, :]
        return s_nearest

    def plan_motion(self, s_src, s_dst):
        q_src = s_src[:self.conf_dim]
        t_src = s_src[-1]
        q_dst = s_dst[:self.conf_dim]
        delta_q = q_dst - q_src
        dist = np.linalg.norm(delta_q)
        nr_time_steps = int(dist/self.max_actuation)
        s_curr = s_src
        states_list = []
        for dt in range(1, nr_time_steps+1):
            alpha = (dt/nr_time_steps)
            q_nxt = q_src * (1-alpha) + q_dst * alpha
            s_nxt = np.hstack((q_nxt, t_src + dt))
            is_collision, _ = self.env.collision_check_transition(s_curr, s_nxt)
            if is_collision:
                break
            else:
                states_list.append(s_nxt)
        return states_list

    def add_states_to_tree(self, state_src: np.ndarray, states: np.ndarray):
        s_src = state_src
        for s_dst in states:
            self.add_edge(s_src, s_dst)
            s_src = s_dst
        self.add_states_to_vertices(states)

    def add_edge(self, s_src: np.ndarray, s_dst: np.ndarray):
        s_src_tuple = tuple(s_src.tolist())
        s_dst_tuple = tuple(s_dst.tolist())
        self.edges[s_src_tuple].append(s_dst_tuple)
        self.parents[s_dst_tuple] = s_src_tuple

    def add_states_to_vertices(self, states: np.ndarray):
        nr_states = states.shape[0]
        i_start = self.verts_cnt
        i_end = self.verts_cnt + nr_states
        if i_end >= self.max_nr_vertices:
            raise RuntimeError("Maximum nr of vertices achieved")
        self.verts[i_start: i_end] = states
        self.verts_cnt += nr_states

    def check_states_if_in_goal_set(self, states: np.ndarray, q_goal: np.ndarray, goal_radii=0.1):
        dists = np.linalg.norm(q_goal - states[:, :self.conf_dim], axis=1)
        is_in_goal = dists <= goal_radii
        state_goal = None
        if is_in_goal.any():
            logger.debug("Found goal")
            i_goal_state = np.argmin(dists)
            state_goal = states[i_goal_state, :]
        smallest_distance_states = dists.min()
        if smallest_distance_states < self.smallest_distance:
            logger.debug("Smallest distance: %s", smallest_distance_states)
            self.smallest_distance = smallest_distance_states
        return state_goal

    def get_path(self, s_src, s_dst):
        s_src_tuple = tuple(s_src.tolist())
        s_dst_tuple = tuple(s_dst.tolist())
        path = [s_src_tuple]
        while s_src_tuple != s_dst_tuple:
            s_src_tuple = self.parents[s_src_tuple]
            path.append(s_src_tuple)
        return path[::-1]


if __name__ == "__main__":
    from dev.mp_time_envs import RobAndRobCollabMPEnv
    from envs.env3d.adversary import Adversary3DOFManipulator
    from envs.env3d.agent import Manipulator3DOF
    from logging import StreamHandler
    import sys


    logger.addHandler(StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    adversary = Adversary3DOFManipulator()
    agent = Manipulator3DOF()
    env_ranges = EnvRanges3D((-1, 1), (-1, 1), (-1, 1))

    render = Matplotlib3D(agent, adversary, env_ranges)
    mpenv = RobAndRobCollabMPEnv(agent, adversary)
    mpenv.reset()
    planner = DynamicRRTPlanner(mpenv)
    q_start = mpenv.agent.sample_random_pose()
    s_start = np.hstack((q_start, 0))
    q_goal = mpenv.agent.sample_random_pose()
    path = planner.plan(s_start, q_goal)


    if path:
        for s in path:
            agent.set_config(s[:planner.conf_dim])
            adversary.set_time(s[-1])
            render.render(goal_state=q_goal, pause_time=1.0)
    # planner.load_state_from_pickle(path_to_solved)
