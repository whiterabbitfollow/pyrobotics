from collections import defaultdict
import logging
import math
import time

import numpy as np

from pyrb.mp.base_world import BaseMPTimeVaryingWorld
from pyrb.mp.planners.shared import PlanningData, Status

logger = logging.Logger(__name__)

def is_vertex_in_goal_region(q, q_goal, goal_region_radius):
    distance = np.linalg.norm(q - q_goal)
    return distance < goal_region_radius


class LocalPlanner:

    def __init__(
            self,
            world,
            min_step_size,
            max_distance,
            global_goal_region_radius,
            max_actuation,
            nr_coll_steps=10
    ):
        self.global_goal_region_radius = global_goal_region_radius
        self.max_distance = max_distance
        self.min_step_size = min_step_size  # TODO: unused...
        self.world = world
        self.max_actuation = max_actuation
        self.nr_coll_steps = nr_coll_steps

    def plan(self, state_src, state_dst, config_global_goal=None, full_plan=False):
        max_nr_steps = self.compute_max_nr_steps(state_src, state_dst, full_plan)
        state_prev = state_src
        config_dst = state_dst[:-1]
        path = np.zeros((max_nr_steps, state_prev.size))
        cnt = 0
        for delta_t in range(1, max_nr_steps + 1):
            config_prev = state_prev[:-1]
            t_prev = state_prev[-1]
            config_delta = np.clip(config_dst - config_prev, -self.max_actuation, self.max_actuation)
            config_nxt = config_prev + config_delta
            state_nxt = np.append(config_nxt, t_prev + 1)
            collision_free_transition = self.world.is_collision_free_transition(
                state_src=state_prev,
                state_dst=state_nxt,
                nr_coll_steps=self.nr_coll_steps
            )
            is_in_global_goal = config_global_goal is not None and is_vertex_in_goal_region(
                config_nxt,
                config_global_goal,
                self.global_goal_region_radius
            )
            if np.abs(config_delta).sum() == 0:
                # TODO: tmp hack.... NEEDS TO BE FIXED!!!
                break
            if collision_free_transition:
                state_prev = state_nxt
                path[cnt, :] = state_nxt
                cnt += 1
            if is_in_global_goal or not collision_free_transition:
                break
        return path[:cnt, :]

    def compute_max_nr_steps(self, state_src, state_dst, full_plan):
        config_src = state_src[:-1]
        config_dst = state_dst[:-1]
        config_delta = config_dst - config_src
        t_src = state_src[-1]
        t_dst = state_dst[-1]
        nr_steps = t_dst - t_src
        distance = np.linalg.norm(config_delta)
        if not full_plan:
            distance = min(distance, self.max_distance)
        nr_steps_full_act = math.ceil(distance / self.max_actuation)
        max_nr_steps = int(min(nr_steps, nr_steps_full_act))
        return max_nr_steps


class RRTPlannerTimeVarying:

    def __init__(
            self,
            world: BaseMPTimeVaryingWorld,
            max_nr_vertices=int(1e4),
            local_planner_max_distance=0.5,
            local_planner_nr_coll_steps=10,
            goal_region_radius=1e-1
    ):
        self.goal_region_radius = goal_region_radius
        self.max_nr_vertices = max_nr_vertices
        self.max_actuation = world.robot.max_actuation
        self.edges_parent_to_children = defaultdict(list)
        time_dimension = 1
        self.vertices = np.zeros((max_nr_vertices, world.robot.nr_joints + time_dimension))
        self.edges_child_to_parent = np.zeros((max_nr_vertices,), dtype=int)
        self.vert_cnt = 0
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,     # TODO: not used.... but better than steps...
            max_distance=local_planner_max_distance,
            global_goal_region_radius=self.goal_region_radius,
            max_actuation=self.max_actuation,
            nr_coll_steps=local_planner_nr_coll_steps
        )

    def clear(self):
        self.vertices.fill(0)
        self.edges_child_to_parent.fill(0)
        self.edges_parent_to_children.clear()
        self.vert_cnt = 0

    def plan(
            self,
            state_start,
            config_goal,
            max_planning_time=np.inf,
            min_planning_time=0,
            time_horizon=300
    ):
        self.add_vertex_to_tree(state_start)
        time_s, time_elapsed = self.start_timer()
        found_solution = False
        while (
                not self.is_tree_full() and
                time_elapsed < max_planning_time and
                not found_solution
        ) or time_elapsed < min_planning_time:
            state_free = self.sample_collision_free_config(time_horizon)
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
                    found_solution = True
                    if time_elapsed < min_planning_time:
                        time_horizon = state_new[-1]
                        print(
                            f"Found solution with time horizon {time_horizon}."
                            f" Min planning time left {min_planning_time - time_elapsed}"
                        )
                    break
                i_parent = i_child
            time_elapsed = time.time() - time_s
        path = self.find_path(state_start, config_goal)
        return path, self.compile_planning_data(path, time_elapsed, self.vert_cnt)

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
            times = self.vertices[:self.vert_cnt:, -1][mask_vertices_goal]
            i_min_time_sub = np.argmin(times)
            min_time = np.min(times)
            i = np.nonzero(mask_vertices_goal)[0][i_min_time_sub]
            state = self.vertices[i, :]
            assert state[-1] == min_time, "Time not correct..."
            path = [state]
            while (state != state_start).any():
                i = self.edges_child_to_parent[i]
                state = self.vertices[i, :]
                path.append(state.ravel())
            path.reverse()
            path = np.vstack(path)
        else:
            path = np.array([]).reshape((0, state_start.size))
        return path

    def sample_collision_free_config(self, time_horizon):
        while True:
            t = np.random.randint(1, time_horizon)
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
            t_closest = np.max(states_valid[:, -1])
            # TODO: could be problematic.... maybe need a time window so that we not only plan one step paths..
            mask_valid_states = self.vertices[:self.vert_cnt, -1] == t_closest
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

    def compile_planning_data(self, path, time_elapsed, nr_verts):
        status = Status.SUCCESS if path.size else Status.FAILURE
        return PlanningData(status=status, time_taken=time_elapsed, nr_verts=nr_verts)


class ModifiedRRTPlannerTimeVarying(RRTPlannerTimeVarying):

    def init_graph_with_all_start_config(self, state_start, time_horizon):
        t_start = int(state_start[-1])
        state_src = state_start.copy()
        i_parent = self.vert_cnt - 1
        for t in range(t_start + 1, time_horizon):
            i_child = self.vert_cnt
            state_dst = state_src.copy()
            state_dst[-1] = t
            if self.world.is_collision_free_transition(state_src=state_src, state_dst=state_dst):
                self.append_vertex(state_dst, i_parent)
                i_parent = i_child
            else:
                break

    def test_naive_solution(self, state_start, config_goal, nr_coll_steps, goal_region_radius):
        state_prev = state_start
        path = []
        while True:
            config_prev = state_prev[:-1]
            t_prev = state_prev[-1]
            config_delta = np.clip(config_goal - config_prev, -self.max_actuation, self.max_actuation)
            config_nxt = config_prev + config_delta
            state_nxt = np.append(config_nxt, t_prev + 1)
            collision_free_transition = self.world.is_collision_free_transition(
                state_src=state_prev,
                state_dst=state_nxt,
                nr_coll_steps=nr_coll_steps
            )
            is_in_global_goal = is_vertex_in_goal_region(
                config_nxt,
                config_goal,
                goal_region_radius
            )
            if collision_free_transition:
                state_prev = state_nxt
                path.append(state_nxt)
            else:
                path.clear()
                break
            if is_in_global_goal:
                break
        path = np.vstack(path) if path else np.array([]).reshape((0,) + state_start.shape)
        return path

    def plan(
            self,
            state_start,
            config_goal,
            max_planning_time=np.inf,
            min_planning_time=0,
            time_horizon=300
    ):
        self.add_vertex_to_tree(state_start)
        self.init_graph_with_all_start_config(state_start, time_horizon)
        naive_path = self.test_naive_solution(
            state_start,
            config_goal,
            nr_coll_steps=self.local_planner.nr_coll_steps,
            goal_region_radius=self.goal_region_radius
        )
        if naive_path.size > 0:
            logger.debug("Naive path solves problem. Done!")
            return naive_path, None
        time_s, time_elapsed = self.start_timer()
        found_solution = False
        while (
                not self.is_tree_full() and
                time_elapsed < max_planning_time and
                not found_solution
        ) or time_elapsed < min_planning_time:
            state_free = self.sample_collision_free_config(time_horizon)
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
                    found_solution = True
                    if time_elapsed < min_planning_time:
                        time_horizon = state_new[-1]
                        logger.debug(
                            "Found solution with time horizon %s. Min planning time left %s",
                            time_horizon, min_planning_time - time_elapsed
                        )
                    else:
                        logger.debug("Found solution with time horizon %s.", time_horizon)
                    break
                i_parent = i_child
            time_elapsed = time.time() - time_s
        path = self.find_path(state_start, config_goal)
        return path, self.compile_planning_data(path, time_elapsed, self.vert_cnt)
