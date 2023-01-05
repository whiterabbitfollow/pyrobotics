from abc import ABCMeta, abstractmethod

import numpy as np

from pyrb.mp.base_world import BaseMPWorld, BaseMPTimeVaryingWorld


class BaseStateSpace(metaclass=ABCMeta):

    def __init__(self, world: BaseMPWorld, dim, limits):
        self.world = world
        self.dim = dim
        self.limits = limits

    @abstractmethod
    def transition_cost(self, state_src, state_dst):
        raise NotImplementedError()

    @abstractmethod
    def find_nearest_state(self, states, state):
        raise NotImplementedError()

    @abstractmethod
    def sample_collision_free_state(self):
        raise NotImplementedError()

    @abstractmethod
    def is_collision_free_state(self, state):
        raise NotImplementedError()

    def is_collision_free_transition(self, state_src, state_dst, min_step_size=None, nr_steps=None):
        if min_step_size is None and nr_steps is None:
            raise RuntimeError("Need to specify either min_step_size or nr_steps")
        if min_step_size is not None:
            dist = np.linalg.norm(state_dst - state_src)
            nr_coll_steps = max(int(dist / min_step_size), 1)
        else:
            nr_coll_steps = nr_steps
        is_collision_free = False
        for i in range(1, nr_coll_steps + 1):
            alpha = i / nr_coll_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            is_collision_free = self.is_collision_free_state(state)
            if not is_collision_free:
                break
        return is_collision_free


class RealVectorStateSpace(BaseStateSpace):

    def is_within_bounds(self, state):
        return ((self.limits[:, 0] <= state) & (state <= self.limits[:, 1])).all()

    def find_nearest_state(self, states, state):
        distance = np.linalg.norm(states - state, axis=1)
        i = np.argmin(distance)
        return i, states[i]

    def get_nearest_states_indices(self, states, state, nearest_radius):
        distances = np.linalg.norm(states - state, axis=1)
        return (distances < nearest_radius).nonzero()[0]

    def linspace(self, state_src, state_dst, nr_points):
        return np.linspace(state_src, state_dst, nr_points)

    def sample_feasible_state(self):
        return np.random.uniform(self.limits[:, 0], self.limits[:, 1])

    def sample_collision_free_state(self):
        while True:
            state = self.sample_feasible_state()
            if self.is_collision_free_state(state):
                return state

    def is_collision_free_state(self, state):
        return self.world.is_collision_free_config(state)

    def distance(self, state_1, state_2):
        return np.linalg.norm(state_1 - state_2)

    def distances(self, state, *, states):
        return np.linalg.norm(state - states, axis=1)

    def transition_cost(self, state_1, state_2):
        return self.distance(state_1, state_2)

    def transition_cost_dst_many(self, state_src, states_dst):
        return np.linalg.norm(state_src - states_dst, axis=1)

    def transition_cost_src_many(self, states_src, state_dst):
        return np.linalg.norm(states_src - state_dst, axis=1)


class RealVectorTimeSpace(BaseStateSpace):

    # (R^n, R_+)
    # Time flows forward

    def __init__(
            self,
            world: BaseMPTimeVaryingWorld,
            dim,
            limits,
            max_time,
            goal_region=None,
            min_time=0,
            gamma=0.1
    ):
        super().__init__(world, dim + 1, limits)
        self.world = world
        self.min_time = min_time
        self.max_time = max_time
        self.goal_region = goal_region
        self.gamma = gamma

    def set_time_interval(self, t_min, t_max):
        self.min_time = t_min
        self.max_time = t_max

    def is_within_bounds(self, state):
        return ((self.limits[:, 0] <= state[:-1]) & (state[:-1] <= self.limits[:, 1])).all() and (self.min_time < state[-1] <= self.max_time)

    def find_nearest_state(self, states, state):
        t = state[-1]
        config = state[:-1]
        mask_valid_states = states[:, -1] < t
        i_state_nearest, state_nearest = None, None
        if mask_valid_states.any():
            gamma = 0.1
            states_valid = states[mask_valid_states]
            distances = np.linalg.norm(states_valid[:, :-1] - config, axis=1) + (t - states_valid[:, -1]) * gamma
            i_vert_mask = np.argmin(distances)
            i_state_nearest = mask_valid_states.nonzero()[0][i_vert_mask]
            state_nearest = states[i_state_nearest]
        return i_state_nearest, state_nearest

    def get_nearest_states_indices(self, states, state, nearest_radius):
        # TODO: deprecated?
        raise NotImplementedError()

    def sample_collision_free_config(self):
        while True:
            config = np.random.uniform(self.limits[:, 0], self.limits[:, 1])
            if self.world.is_collision_free_config(config):
                return config

    def sample_collision_free_state(self):
        time_horizon = self.max_time
        if self.goal_region is not None and np.isfinite(self.goal_region.time_horizon):
            time_horizon = self.goal_region.time_horizon
        while True:
            t_low, t_upper = self.min_time + 1, time_horizon + 1
            t = t_low if t_low == t_upper else np.random.randint(t_low, t_upper)
            # TODO: should constrain sampling based on t... actuation etc.
            config = np.random.uniform(self.limits[:, 0], self.limits[:, 1])
            state = np.append(config, t)
            self.world.set_time(t)
            if self.world.is_collision_free_config(config):
                return state

    def is_collision_free_state(self, state):
        config = state[:-1]
        t = state[-1]
        self.world.set_time(t)
        return self.world.is_collision_free_config(config)

    def distance(self, state_1, state_2):
        return np.linalg.norm(state_1[:-1] - state_2[:-1])

    def distances(self, state, *, states):
        return np.linalg.norm(state[:-1] - states[:, :-1], axis=1)

    def transition(self, t, dt=1):
        return min(t + dt, self.max_time)

    def detransition(self, t, dt=1):
        return max(t - dt, 0)

    def linspace(self, state_src, state_dst, nr_points):
        return np.linspace(state_src, state_dst, nr_points)

    def get_indices_of_states_within_time(self, config, states, t, nearest_radius):
        mask = (states[:, -1] == t)
        valid_time_n_distance_verts_indxs = np.array([], dtype=int)
        if mask.any():
            valid_time_verts_indxs = mask.nonzero()[0]
            distances = np.linalg.norm(states[mask, :-1] - config, axis=1)
            valid_time_n_distance_verts_indxs = valid_time_verts_indxs[(distances < nearest_radius)]
        return valid_time_n_distance_verts_indxs

    def is_valid_time_direction(self, t_src, t_dst):
        return t_dst > t_src

    def transition_cost(self, state_src, state_dst):
        nr_time_steps = np.abs(state_dst[-1] - state_src[-1]) * self.gamma
        if self.goal_region and self.goal_region.is_config_within(state_dst):
            return nr_time_steps
        distance = np.linalg.norm(state_src[:-1] - state_dst[:-1])
        cost = distance + nr_time_steps
        return cost

    def transition_cost_src_many(self, states_src, state_dst):
        nr_time_steps = np.abs(state_dst[-1] - states_src[:, -1]) * self.gamma
        if self.goal_region and self.goal_region.is_config_within(state_dst):
            return nr_time_steps
        distances = np.linalg.norm(states_src[:, :-1] - state_dst[:-1], axis=1)
        costs = distances + nr_time_steps
        return costs

    def transition_cost_dst_many(self, state_src, states_dst):
        n = states_dst.shape[0]
        mask_within = self.goal_region.are_configs_within(states_dst) if self.goal_region else np.zeros(n).astype(bool)
        mask = ~mask_within
        nr_time_steps = np.abs(states_dst[:, -1] - state_src[-1]) * self.gamma
        distances = np.linalg.norm(state_src[:-1] - states_dst[mask, :-1], axis=1)
        costs = nr_time_steps
        costs[mask] += distances
        return costs


class RealVectorPastTimeSpace(RealVectorTimeSpace):
    # (R^n, R_+)
    # Time flows backwards
    def __init__(self, world, dim, limits, max_time, goal_region=None, min_time=0, gamma=.1):
        self.world = world
        self.dim = dim + 1
        self.limits = limits
        self.max_time = max_time
        self.min_time = min_time
        self.goal_region = goal_region
        self.gamma = gamma

    def find_nearest_state(self, states, state):
        t = state[-1]
        config = state[:-1]
        mask_valid_states = states[:, -1] > t
        i_state_nearest, state_nearest = None, None
        if mask_valid_states.any():
            states_valid = states[mask_valid_states]
            gamma = 0.1
            distance = np.linalg.norm(states_valid[:, :-1] - config, axis=1) + (states_valid[:, -1] - t) * gamma
            i_vert_mask = np.argmin(distance)
            i_state_nearest = mask_valid_states.nonzero()[0][i_vert_mask]
            state_nearest = states[i_state_nearest]
        return i_state_nearest, state_nearest

    def transition(self, t, dt=1):
        return max(t - dt, 0)

    def detransition(self, t, dt=1):
        return min(t + dt, self.max_time)

    def is_valid_time_direction(self, t_src, t_dst):
        return t_dst < t_src
