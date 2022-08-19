import numpy as np

from pyrb.mp.planners.moving.local_planners import TimeModes
from pyrb.mp.planners.static.rrt import Tree


class TreeForwardTime(Tree):

    def __init__(self, *args, **kwargs):
        self.time_mode = TimeModes.FORWARD
        super().__init__(*args, **kwargs)

    def find_nearest_vertex(self, state):
        t = state[-1]
        config = state[:-1]
        mask_valid_states = self.vertices[:self.vert_cnt, -1] < t
        i_vert, vert = None, None
        if mask_valid_states.any():
            states_valid = self.vertices[:self.vert_cnt][mask_valid_states]
            # t_closest = np.max(states_valid[:, -1]) # TODO: Not sure this is optimal
            # # TODO: could be problematic.... maybe need a time window so that we not only plan one step paths..
            # mask_valid_states = self.vertices[:self.vert_cnt, -1] == t_closest
            # states_valid = self.vertices[:self.vert_cnt][mask_valid_states]
            distance = np.linalg.norm(states_valid[:, :-1] - config, axis=1)
            i_vert_mask = np.argmin(distance)
            i_vert = mask_valid_states.nonzero()[0][i_vert_mask]
            vert = self.vertices[i_vert]
        return i_vert, vert


class TreeBackwardTime(Tree):

    def __init__(self, *args, **kwargs):
        self.time_mode = TimeModes.BACKWARD
        super().__init__(*args, **kwargs)

    def find_nearest_vertex(self, state):
        t = state[-1]
        config = state[:-1]
        mask_valid_states = self.vertices[:self.vert_cnt, -1] > t
        i_vert, vert = None, None
        if mask_valid_states.any():
            states_valid = self.vertices[:self.vert_cnt][mask_valid_states]
            # t_closest = np.max(states_valid[:, -1]) # TODO: Not sure this is optimal
            # # TODO: could be problematic.... maybe need a time window so that we not only plan one step paths..
            # mask_valid_states = self.vertices[:self.vert_cnt, -1] == t_closest
            # states_valid = self.vertices[:self.vert_cnt][mask_valid_states]
            distance = np.linalg.norm(states_valid[:, :-1] - config, axis=1)
            i_vert_mask = np.argmin(distance)
            i_vert = mask_valid_states.nonzero()[0][i_vert_mask]
            vert = self.vertices[i_vert]
        return i_vert, vert