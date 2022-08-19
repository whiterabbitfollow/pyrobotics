import time
import logging

import numpy as np

from pyrb.mp.base_world import BaseMPTimeVaryingWorld
from pyrb.mp.planners.moving.time_optimal.tree import TreeForwardTime, TreeBackwardTime
from pyrb.mp.utils.utils import start_timer
from pyrb.mp.planners.moving.local_planners import LocalPlannerRRTConnect
from pyrb.mp.planners.static.local_planners import LocalPlannerStatus

logger = logging.Logger(__name__)


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

        self.tree_start = TreeForwardTime(max_nr_vertices, vertex_dim=world.robot.nr_joints + time_dim)
        self.tree_goal = TreeBackwardTime(max_nr_vertices, vertex_dim=world.robot.nr_joints + time_dim)

        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlannerRRTConnect(
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
        time_s, time_elapsed = start_timer()
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
        path_state_to_start = self.tree_start.find_path_to_root_from_vertex_index(i_state_start)
        path_state_to_goal = self.tree_goal.find_path_to_root_from_vertex_index(i_state_goal)
        return np.vstack([path_state_to_start[::-1, :], path_state_to_goal[1:, :]])

    def ingest_path_in_tree(self, tree, path, i_src):
        i_parent = i_src
        for state_new in path:
            i_child = tree.vert_cnt
            tree.append_vertex(state_new, i_parent=i_parent)
            i_parent = i_child

    def sample_collision_free_config(self, time_horizon):
        while True:
            t = np.random.randint(1, time_horizon)
            # TODO: should constrain sampling based on t... actuation etc.
            config = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            state = np.append(config, t)
            if self.world.is_collision_free_state(state):
                return state
