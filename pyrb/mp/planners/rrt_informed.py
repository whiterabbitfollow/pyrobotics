import logging

import numpy as np

from pyrb.mp.planners.rrt import RRTPlanner


RRT_PLANNER_LOGGER_NAME = __file__
logger = logging.getLogger(RRT_PLANNER_LOGGER_NAME)


def initialize_ellipsoid(x_start, x_goal):
    x_center = (x_start + x_goal) / 2
    x_dir = x_goal - x_start
    c_min = np.linalg.norm(x_dir)
    a_1 = (x_dir / c_min).reshape(-1, 1)
    id_1 = np.zeros((a_1.size, 1))
    id_1[0, 0] = 1
    M = a_1 @ id_1.T
    U, s, V_t = np.linalg.svd(M)
    diag_vals = np.append(np.ones((s.size - 1,)), np.linalg.det(U) * np.linalg.det(V_t))
    C = U @ np.diag(diag_vals) @ V_t
    c_max = np.inf
    return x_center, c_min, c_max, C


class RRTInformedPlanner(RRTPlanner):

    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_center = None
        self.c_min = None
        self.c_max = None
        self.C = None
        self.CL = None

    def initialize_planner(self, state_start, goal_region):
        super().initialize_planner(state_start, goal_region)
        self.x_center, self.c_min, self.c_max, self.C = initialize_ellipsoid(state_start, goal_region.state)

    def report_goal_state(self, i_new):
        super().report_goal_state(i_new)
        path = self.tree.find_path_to_root_from_vertex_index(i_new)
        path = path[::-1]
        c_max = self.c_max
        c = np.linalg.norm(path[1:] - path[:-1], axis=1).sum() + np.linalg.norm(self.goal_region.state - path[-1])
        self.c_max = min(c, c_max)
        if self.c_max != c_max:
            r = np.zeros((self.space.dim,))
            r[0] = self.c_max / 2
            assert self.c_max > self.c_min
            r[1:] = np.sqrt(self.c_max ** 2 - self.c_min ** 2) / 2
            L = np.diag(r)
            self.CL = self.C @ L
            logger.debug("Cost improved, old: %s, new: %s", self.c_max, c_max)

    def sample(self):
        if self.c_max < np.inf:
            x_ball = self.sample_unit_ball(self.space.dim)
            x_rand = self.CL @ x_ball + self.x_center
            return x_rand
        else:
            return super().sample()

    @staticmethod
    def sample_unit_ball(n):
        while True:
            x = np.random.uniform(-1, 1, n)
            if np.linalg.norm(x) < 1:
                break
        return x
