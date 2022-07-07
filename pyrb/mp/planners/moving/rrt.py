import time

import numpy as np
from collections import defaultdict


from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.shared import PlanningData, Status


def is_vertex_in_goal_region(q, q_goal, goal_region_radius):
    distance = np.linalg.norm(q - q_goal)
    return distance < goal_region_radius


class LocalPlanner:

    def __init__(self, world, min_step_size, max_distance, global_goal_region_radius, max_actuation):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size
        self.world = world
        self.max_actuation = max_actuation

    def plan(self, state_src, state_dst, config_global_goal):
        # assumes state_src is collision free
        config_src = state_src[:-1]
        config_dst = state_dst[:-1]
        config_delta = config_dst - config_src
        t_src = state_src[-1]
        t_dst = state_dst[-1]
        nr_steps = int(t_dst - t_src)
        distance = min(self.max_distance, np.linalg.norm(config_delta))
        nr_steps_full_act = int(distance / self.max_actuation)
        max_nr_steps = max(nr_steps, nr_steps_full_act)
        t_prev = t_src
        state = state_src
        path = np.zeros((nr_steps, state.size))
        cnt = 0
        for i in range(1, int(nr_steps) + 1):
            alpha = i / max_nr_steps
            config_nxt = config_dst * alpha + (1 - alpha) * config_src
            state_nxt = np.append(config_nxt, t_prev + 1)
            collision_free_transition = self.world.is_collision_free_transition(state_src=state, state_dst=state_nxt)
            is_in_global_goal = is_vertex_in_goal_region(config_nxt, config_global_goal, self.global_goal_region_radius)
            if collision_free_transition:
                path[cnt, :] = state_nxt
                cnt += 1
                t_prev += 1
            if is_in_global_goal or not collision_free_transition:
                break
        return path[:cnt, :]

    def is_transition_coll_free(self, state_src, state_dst):
        state_delta = state_dst - state_src
        distance = np.linalg.norm(state_delta)
        nr_steps = int(distance / self.min_step_size)
        collision_free_transition = False
        for i in range(1, nr_steps + 1):
            alpha = i / nr_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            collision_free_transition = self.world.is_collision_free_state(state)
            if not collision_free_transition:
                break
        return collision_free_transition


class RRTPlannerTimeVarying:

    def __init__(
            self,
            world: BaseMPWorld,
            time_horizon,
            max_actuation,
            max_nr_vertices=int(1e4),
            max_distance_local_planner=0.5,
    ):
        self.time_horizon = time_horizon
        self.max_nr_vertices = max_nr_vertices
        self.edges_parent_to_children = defaultdict(list)
        self.vertices = np.zeros((max_nr_vertices, world.robot.nr_joints + 1))
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.max_distance_local_planner = max_distance_local_planner
        self.vert_cnt = 0
        self.goal_region_radius = 1e-1
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,
            max_distance=max_distance_local_planner,
            global_goal_region_radius=self.goal_region_radius,
            max_actuation=max_actuation
        )

    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.edges_parent_to_children.clear()
        self.vert_cnt = 0

    def plan(self, state_start, config_goal, max_planning_time=np.inf):
        self.clear()
        self.add_vertex_to_tree(state_start)
        path = np.array([])
        time_s, time_elapsed = self.start_timer()
        while not self.is_tree_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.find_nearest_vertex(state_free)
            if i_nearest is None:
                continue
            local_path = self.local_planner.plan(state_nearest, state_free, config_goal)
            i_parent = i_nearest
            for state_new in local_path:
                i_child = self.vert_cnt
                self.append_vertex(state_new, i_parent=i_parent)
                config_new = state_new[:-1]
                if is_vertex_in_goal_region(config_new, config_goal, self.goal_region_radius):
                    path = self.find_path(state_start, config_goal)
                    break
                i_parent = i_child
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)

    def start_timer(self):
        time_s = time.time()
        time_elapsed = time.time() - time_s
        return time_s, time_elapsed

    def is_tree_full(self):
        return self.vert_cnt >= self.max_nr_vertices

    def find_path(self, state_start, config_goal):
        distances = np.linalg.norm(config_goal - self.vertices[:self.vert_cnt, :-1], axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            i = mask_vertices_goal.nonzero()[0]
            state = self.vertices[i, :]
            path = [state]
            while (state != state_start).any():
                i = self.edges_child_to_parent[i]
                state = self.vertices[i, :]
                path.append(state.ravel())
            path.reverse()
            path = np.vstack(path)
        else:
            path = np.array([])
        return path

    def sample_collision_free_config(self):
        while True:
            t = np.random.randint(1, self.time_horizon)
            # TODO: should constrain sampling based on t... actuation etc.
            config = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            state = np.append(config, t)
            if self.world.is_collision_free_state(state):
                return state

    def find_nearest_vertex(self, state):
        t = state[-1]
        config = state[:-1]
        mask_valid_states = self.vertices[:self.vert_cnt, -1] < t
        i_vert, vert = None, None
        if mask_valid_states.any():
            states_valid = self.vertices[:self.vert_cnt][mask_valid_states]
            distance = np.linalg.norm(states_valid[:, :-1] - config, axis=1)
            i_vert_mask = np.argmin(distance)
            i_vert = mask_valid_states.nonzero()[0][i_vert_mask]
            vert = self.vertices[i_vert]
        return i_vert, vert

    def insert_vertex_in_tree(self, i, state):
        self.vertices[i, :] = state

    def append_vertex(self, state, i_parent):
        self.vertices[self.vert_cnt, :] = state
        self.create_edge(i_parent, self.vert_cnt)
        self.vert_cnt += 1

    def create_edge(self, i_parent, i_child):
        self.edges_parent_to_children[i_parent].append(i_child)
        self.edges_child_to_parent[i_child] = i_parent

    def add_vertex_to_tree(self, state):
        self.vertices[self.vert_cnt, :] = state
        self.vert_cnt += 1

    def compile_planning_data(self, path, time_elapsed):
        status = Status.SUCCESS if path.size else Status.FAILURE
        return PlanningData(status=status, time_taken=time_elapsed)
