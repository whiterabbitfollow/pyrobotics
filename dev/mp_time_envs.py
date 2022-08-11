import numpy as np


class RobAndRobCollabMPEnv:

    def __init__(self, agent, adversary):
        self.adversary = adversary
        self.agent = agent
        self.time_step = 0

    def sample_collision_free_config(self):
        while True:
            state = self.agent.sample_random_pose()
            self.agent.set_config(state)
            if not self.is_collision():
                break
        return state

    def reset(self):
        self.adversary.reset()
        self.set_time(0)

    def clear(self):
        self.set_time(0)

    def is_collision(self):
        return self.agent.collision_manager.in_collision_other(self.adversary.collision_manager)

    def set_time(self, t):
        if self.time_step == t and self.adversary.time_step == t:
            return
        self.time_step = t
        self.adversary.set_time(self.time_step)

    def get_min_collision_distance(self):
        return self.agent.collision_manager.min_distance_other(self.adversary.collision_manager)

    def collision_check_transition(self, state_from: np.ndarray, state_to: np.ndarray):
        nr_time_steps = 10
        states = np.linspace(state_from, state_to, nr_time_steps + 1)   # TODO: depends on the scale
        is_collision = True
        state_coll_free = state_from    # Assume collision free
        for t, s in enumerate(states[1:], 1):
            dt = t/nr_time_steps
            self.adversary.set_time(self.time_step + dt)
            self.agent.set_config(s)
            is_collision = self.is_collision()
            if is_collision:
                break
            state_coll_free = s
        self.agent.set_config(state_coll_free)
        return is_collision, state_coll_free

    def collision_check_transition_with_distance(self, state_from, state_to):
        nr_time_steps = 10
        states = np.linspace(state_from, state_to, nr_time_steps + 1, endpoint=True)  # TODO: depends on the scale
        is_collision = True
        state_coll_free = state_from  # Assume collision free
        min_distance = self.get_min_collision_distance()
        for t, s in enumerate(states[1:], 1):
            dt = t/nr_time_steps
            self.adversary.set_time(self.time_step + dt)
            self.agent.set_config(s)
            is_collision = self.is_collision()
            if is_collision:
                min_distance = 0
                break
            min_distance = min(self.get_min_collision_distance(), min_distance)
            state_coll_free = s
        self.agent.set_config(state_coll_free)
        return is_collision, state_coll_free, min_distance

    def grid_collision_free_vertices(self, t, return_distance=False):
        self.set_time(t)
        verts = self.get_mesh_grid()
        coll_free_configs = []
        distances = {}
        for s in verts:
            self.agent.set_config(s)
            if not self.is_collision():
                coll_free_configs.append(s)
                if return_distance:
                    distances[(tuple(s), t)] = self.get_min_collision_distance()

        if return_distance:
            return coll_free_configs, distances
        else:
            return coll_free_configs

    def get_mesh_grid(self):
        grids = []
        for joint_nr in range(self.agent.nr_joints):
            grid_joint = np.arange(
                self.agent.joint_limits[joint_nr, 0],
                self.agent.joint_limits[joint_nr, 1],
                self.agent.max_actuation
            )
            grids.append(grid_joint)
        points = np.meshgrid(*grids)
        return np.c_[[p.ravel() for p in points]].T

    def is_state_collision_free(self, t: float, state):
        self.set_time(t)
        self.agent.set_config(state)
        return not self.is_collision()

    def is_edge_free(self, t: float, state_from, state_to):
        self.set_time(t)
        is_collision, _ = self.collision_check_transition(state_from, state_to)
        return not is_collision

    def get_edge_collision_distance(self, t: float, state_from, state_to):
        self.set_time(t)
        is_collision, _, distance = self.collision_check_transition_with_distance(state_from, state_to)
        return not is_collision, distance

    def get_state_collision_distance(self, t: float, state):
        self.set_time(t)
        self.agent.set_config(state)
        return self.get_min_collision_distance()

    def is_action_valid(self, action):
        return (-self.agent.max_actuation <= action).all() & (action <= self.agent.max_actuation).all()

    def is_state_valid(self, state):
        return (self.agent.joint_limits[:, 0] <= state).all() & (state <= self.agent.joint_limits[:, 1]).all()

    def sample_collision_free_vertices_around_state(self, state, t: float, nr_points=10):
        # TODO: not perfect....
        # TODO: not collision free?
        self.set_time(t)
        size = (nr_points, self.agent.nr_joints)
        radius = self.agent.max_actuation
        state_lower = np.clip(state - radius, self.agent.joint_limits[:, 0], self.agent.joint_limits[:, 1])
        state_upper = np.clip(state + radius, self.agent.joint_limits[:, 0], self.agent.joint_limits[:, 1])
        states = np.random.uniform(state_lower, state_upper, size)
        collision_free_states = []
        for s in states:
            self.agent.set_config(s)
            if not self.is_collision():
                collision_free_states.append(s)
        return np.array(collision_free_states)

    def sample_collision_free_vertices(self, t: float, nr_points: int):
        self.set_time(t)
        coll_free_configs = []
        size = (nr_points, self.agent.nr_joints)
        while len(coll_free_configs) < nr_points:
            states = np.random.uniform(self.agent.joint_limits[:, 0], self.agent.joint_limits[:, 1], size)
            for s in states:
                self.agent.set_config(s)
                if not self.is_collision():
                    coll_free_configs.append(s)
        return coll_free_configs

    def get_system_params(self):
        return {"traj": self.adversary.traj}

    def set_system_params(self, *, traj):
        self.adversary.traj = traj
