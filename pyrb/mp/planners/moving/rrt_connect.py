import time
import logging
import math

import numpy as np

from pyrb.mp.base_world import BaseMPTimeVaryingWorld
from pyrb.mp.planners.shared import PlanningData, Status
from pyrb.mp.planners.static.rrt import Tree
from pyrb.mp.planners.static.rrt_connect import LocalPlannerStatus

logger = logging.Logger(__name__)


from enum import Enum, auto


class TimeModes(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class TreeStart(Tree):

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


class TreeGoal(Tree):

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

    def plan(self, state_src, state_dst, time_mode, state_global_goal=None, full_plan=False):
        state_prev = state_src
        config_dst = state_dst[:-1]

        max_nr_steps = math.ceil(self.max_distance/self.max_actuation) if not full_plan else np.inf
        path = []
        collision_free_transition = True
        has_reached_dst = False
        step_nr = 0
        is_passed_time_horizon = False

        t_dst = state_dst[-1]

        while step_nr < max_nr_steps and collision_free_transition and not has_reached_dst and not is_passed_time_horizon:
            config_prev = state_prev[:-1]
            t_prev = state_prev[-1]
            config_delta = np.clip(config_dst - config_prev, -self.max_actuation, self.max_actuation)
            config_nxt = config_prev + config_delta
            if time_mode == TimeModes.FORWARD:
                t_nxt = t_prev + 1
            else:
                t_nxt = t_prev - 1
            state_nxt = np.append(config_nxt, t_nxt)
            collision_free_transition = self.world.is_collision_free_transition(
                state_src=state_prev,
                state_dst=state_nxt,
                nr_coll_steps=self.nr_coll_steps
            )

            # is_in_global_goal = state_global_goal is not None and is_vertex_in_goal_region(
            #     config_nxt,
            #     state_global_goal[:-1],
            #     self.global_goal_region_radius
            # ) and t_nxt == state_global_goal[-1]

            if collision_free_transition:
                state_prev = state_nxt
                path.append(state_nxt)
            else:
                path.clear()

            step_nr += 1
            has_reached_dst = np.isclose(state_nxt, state_dst).all()

            if time_mode == TimeModes.FORWARD:
                is_passed_time_horizon = t_nxt >= t_dst
            else:
                is_passed_time_horizon = t_nxt <= t_dst

        if collision_free_transition and has_reached_dst:
            status = LocalPlannerStatus.REACHED
        elif collision_free_transition and not has_reached_dst:
            status = LocalPlannerStatus.ADVANCED
        else:
            status = LocalPlannerStatus.TRAPPED
        path = np.vstack(path) if path else np.array([]).reshape((0, state_dst.size))
        return status, path


class RRTConnectPlannerTimeVarying:

    def __init__(
            self,
            world: BaseMPTimeVaryingWorld,
            max_nr_vertices=int(1e4),
            local_planner_max_distance=0.5,
            local_planner_nr_coll_steps=10,
            goal_region_radius=1e-1
    ):
        self.goal_region_radius = goal_region_radius
        self.max_actuation = world.robot.max_actuation
        time_dim = 1

        self.tree_start = TreeStart(max_nr_vertices, vertex_dim=world.robot.nr_joints + time_dim)
        self.tree_goal = TreeGoal(max_nr_vertices, vertex_dim=world.robot.nr_joints + time_dim)

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
        self.tree_start.clear()
        self.tree_goal.clear()

    def plan(
            self,
            state_start,
            state_goal,
            max_planning_time=np.inf
    ):
        self.tree_start.add_vertex(state_start)
        self.tree_goal.add_vertex(state_goal)

        time_horizon = state_goal[-1]
        time_s, time_elapsed = self.start_timer()
        found_solution = False

        tree_a = self.tree_start
        tree_b = self.tree_goal
        path = None
        while (
                not self.tree_start.is_full() and
                time_elapsed < max_planning_time and
                not found_solution and path is None
        ):
            path = self.rrt_connect(tree_a, tree_b, time_horizon)
            tree_a, tree_b = self.swap_trees(tree_a, tree_b)
            time_elapsed = time.time() - time_s
        path = np.array([]).reshape((0, state_goal.size)) if path is None else path
        return path, None

    def rrt_connect(self, tree_a, tree_b, time_horizon):
        path = None
        state_free = self.sample_collision_free_config(time_horizon)
        i_nearest_a, state_nearest_a = tree_a.find_nearest_vertex(state_free)

        if i_nearest_a is None:
            return path

        status, local_path = self.local_planner.plan(
            state_nearest_a,
            state_free,
            time_mode=tree_a.time_mode
        )
        if status == LocalPlannerStatus.TRAPPED:
            return path

        self.ingest_path_in_tree(tree_a, local_path, i_nearest_a)
        i_state_new_a = tree_a.vert_cnt - 1

        state_new_a = local_path[-1, :]
        i_nearest_b, state_nearest_b = tree_b.find_nearest_vertex(state_new_a)

        status, local_path = self.local_planner.plan(
            state_nearest_b,
            state_new_a,
            full_plan=True,
            time_mode=tree_b.time_mode
        )

        if status == LocalPlannerStatus.REACHED:
            self.ingest_path_in_tree(tree_b, local_path, i_nearest_b)
            i_state_new_b = tree_b.vert_cnt - 1
            i_state_start, i_state_goal = self.sort_indices(tree_a, i_state_new_a, i_state_new_b)
            path = self.connect_trees(
                i_state_start,
                i_state_goal
            )
        return path

    def sort_indices(self, tree_a, i_a, i_b):
        is_tree_a_start_tree = tree_a == self.tree_start
        if is_tree_a_start_tree:
            i_state_start = i_a
            i_state_goal = i_b
        else:
            i_state_start = i_b
            i_state_goal = i_a
        return i_state_start, i_state_goal

    def swap_trees(self, tree_a, tree_b):
        return tree_b, tree_a

    def connect_trees(self, i_state_start, i_state_goal):
        path_state_to_start = self.find_path_to_root_from_vertex_index(self.tree_start, i_state_start)
        path_state_to_goal = self.find_path_to_root_from_vertex_index(self.tree_goal, i_state_goal)
        return np.vstack([path_state_to_start[::-1, :], path_state_to_goal[1:, :]])

    def find_path_to_root_from_vertex_index(self, tree, i_vert):
        vertices = tree.get_vertices()
        state = vertices[i_vert, :]
        path = [state]
        while i_vert:
            i_vert = tree.get_vertex_parent_index(i_vert)
            state = vertices[i_vert, :]
            path.append(state)
        return np.vstack(path)

    def ingest_path_in_tree(self, tree, path, i_src):
        i_parent = i_src
        for state_new in path:
            i_child = tree.vert_cnt
            tree.append_vertex(state_new, i_parent=i_parent)
            i_parent = i_child

    def start_timer(self):
        time_s = time.time()
        time_elapsed = time.time() - time_s
        return time_s, time_elapsed

    # def find_path(self, state_start, config_goal):
    #     distances = np.linalg.norm(config_goal - self.vertices[:self.vert_cnt, :-1], axis=1)
    #     mask_vertices_goal = distances < self.goal_region_radius
    #     if mask_vertices_goal.any():
    #         times = self.vertices[:self.vert_cnt:, -1][mask_vertices_goal]
    #         i_min_time_sub = np.argmin(times)
    #         min_time = np.min(times)
    #         i = np.nonzero(mask_vertices_goal)[0][i_min_time_sub]
    #         state = self.vertices[i, :]
    #         assert state[-1] == min_time, "Time not correct..."
    #         path = [state]
    #         while (state != state_start).any():
    #             i = self.edges_child_to_parent[i]
    #             state = self.vertices[i, :]
    #             path.append(state.ravel())
    #         path.reverse()
    #         path = np.vstack(path)
    #     else:
    #         path = np.array([]).reshape((0, state_start.size))
    #     return path

    def sample_collision_free_config(self, time_horizon):
        while True:
            t = np.random.randint(1, time_horizon)
            # TODO: should constrain sampling based on t... actuation etc.
            config = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            state = np.append(config, t)
            if self.world.is_collision_free_state(state):
                return state

    def compile_planning_data(self, path, time_elapsed, nr_verts):
        status = Status.SUCCESS if path.size else Status.FAILURE
        return PlanningData(status=status, time_taken=time_elapsed, nr_verts=nr_verts)
