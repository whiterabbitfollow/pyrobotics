import logging
import time

import numpy as np

from pyrb.mp.base_world import BaseMPTimeVaryingWorld
from pyrb.mp.planners.moving.local_planners import LocalPlanner
from pyrb.mp.planners.moving.time_optimal.tree import TreeForwardTime
from pyrb.mp.utils.utils import is_vertex_in_goal_region, start_timer, compile_planning_data

logger = logging.Logger(__name__)


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
        time_dim = 1
        self.tree = TreeForwardTime(max_nr_vertices=max_nr_vertices, vertex_dim=world.robot.nr_joints + time_dim)
        self.max_actuation = world.robot.max_actuation
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
        self.tree.clear()

    def plan(
            self,
            state_start,
            config_goal,
            max_planning_time=np.inf,
            min_planning_time=0,
            time_horizon=300
    ):
        self.tree.add_vertex(state_start)
        time_s, time_elapsed = start_timer()
        found_solution = False
        while (
                not self.tree.is_full() and
                time_elapsed < max_planning_time and
                not found_solution
        ) or time_elapsed < min_planning_time:
            state_free = self.sample_collision_free_config(time_horizon)
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            if i_nearest is None:
                continue
            local_path = self.local_planner.plan(state_nearest, state_free, config_goal)
            i_parent = i_nearest
            for state_new in local_path:
                i_child = self.tree.vert_cnt
                self.tree.append_vertex(state_new, i_parent=i_parent)
                config_new = state_new[:-1]
                if is_vertex_in_goal_region(config_new, config_goal, self.goal_region_radius):
                    found_solution = True
                    if time_elapsed < min_planning_time:
                        time_horizon = state_new[-1]
                        logger.debug(
                            f"Found solution with time horizon {time_horizon}."
                            f" Min planning time left {min_planning_time - time_elapsed}"
                        )
                    break
                i_parent = i_child
            time_elapsed = time.time() - time_s
        path = self.find_path(state_start, config_goal)
        return path, compile_planning_data(path, time_elapsed, self.tree.vert_cnt)

    def find_path(self, state_start, config_goal):
        vertices = self.tree.get_vertices()
        distances = np.linalg.norm(config_goal - vertices[:, :-1], axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            times = vertices[:, -1][mask_vertices_goal]
            i_min_time_sub = np.argmin(times)
            i = np.nonzero(mask_vertices_goal)[0][i_min_time_sub]
            path = self.tree.find_path_to_root_from_vertex_index(i)
            path = path[::-1]
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


class ModifiedRRTPlannerTimeVarying(RRTPlannerTimeVarying):

    def init_graph_with_all_start_config(self, state_start, time_horizon):
        t_start = int(state_start[-1])
        state_src = state_start.copy()
        i_parent = self.tree.vert_cnt - 1
        for t in range(t_start + 1, time_horizon):
            i_child = self.tree.vert_cnt
            state_dst = state_src.copy()
            state_dst[-1] = t
            if self.world.is_collision_free_transition(state_src=state_src, state_dst=state_dst):
                self.tree.append_vertex(state_dst, i_parent)
                i_parent = i_child
            else:
                break

    def test_naive_solution(self, state_start, config_goal, nr_coll_steps, goal_region_radius):
        state_prev = state_start
        path = [state_start]
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
        self.tree.add_vertex(state_start)
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
        time_s, time_elapsed = start_timer()
        found_solution = False
        while (
                not self.tree.is_full() and
                time_elapsed < max_planning_time and
                not found_solution
        ) or time_elapsed < min_planning_time:
            state_free = self.sample_collision_free_config(time_horizon)
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            if i_nearest is None:
                continue
            local_path = self.local_planner.plan(state_nearest, state_free, config_goal)
            i_parent = i_nearest
            for state_new in local_path:
                i_child = self.tree.vert_cnt
                self.tree.append_vertex(state_new, i_parent=i_parent)
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
        return path, compile_planning_data(path, time_elapsed, self.tree.vert_cnt)
