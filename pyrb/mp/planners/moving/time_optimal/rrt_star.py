import logging
import time

import numpy as np

from pyrb.mp.base_world import BaseMPTimeVaryingWorld
from pyrb.mp.planners.moving.local_planners import LocalPlanner
from pyrb.mp.planners.moving.time_optimal.tree import TreeForwardTimeRewire
from pyrb.mp.utils.utils import start_timer, compile_planning_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStarPlannerTimeVarying:

    def __init__(
            self,
            world: BaseMPTimeVaryingWorld,
            max_nr_vertices=int(1e4),
            local_planner_max_distance=0.5,
            local_planner_nr_coll_steps=10,
            goal_region_radius=1e-1,
            nearest_radius=.2
    ):
        self.goal_region_radius = goal_region_radius
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,  # TODO: not used.... but better than steps...
            max_distance=local_planner_max_distance,
            global_goal_region_radius=self.goal_region_radius,
            max_actuation=world.robot.max_actuation,
            nr_coll_steps=local_planner_nr_coll_steps
        )
        time_dim = 1
        self.tree = TreeForwardTimeRewire(
            local_planner=self.local_planner,
            max_nr_vertices=max_nr_vertices,
            nearest_radius=nearest_radius,
            vertex_dim=world.robot.nr_joints + time_dim
        )
        self.ingested_vertices = []
        # TODO: add a list of newest vertices that we can evaluate at the end if they are in goal
        # TODO: could add a cone around sampling space parameterized by time.

    def clear(self):
        self.tree.clear()
        self.ingested_vertices.clear()

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
            # TODO: Need some better stopping criteria, that is configurable
            state_free = self.sample_collision_free_config(time_horizon)
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, config_goal)
            if local_path.size:
                i_start = self.tree.vert_cnt
                for state_new in local_path:
                    # TODO: do we need to rewire every state in the local_path???
                    i_new = self.tree.vert_cnt
                    self.tree.add_vertex(state_new)
                    self.tree.rewire(i_new, state_new)
                ingested_verts = self.tree.vertices[i_start: self.tree.vert_cnt, :]
                ingested_configs = ingested_verts[:, :-1]
                ingested_times = ingested_verts[:, -1]
                distance = np.linalg.norm(ingested_configs - config_goal, axis=1)
                in_goal = distance < self.goal_region_radius
                if in_goal.any():
                    time_horizon = min(time_horizon, ingested_times[in_goal].min())
                    logger.debug(
                        f"Found solution, "
                        f"time horizon: {time_horizon}, "
                        f"planning time left: {max_planning_time - time_elapsed}"
                    )
                    found_solution = True
            time_elapsed = time.time() - time_s
        path = self.find_path(state_start, config_goal)
        return path, compile_planning_data(path, time_elapsed, self.tree.vert_cnt)

    def sample_collision_free_config(self, time_horizon):
        while True:
            t = np.random.randint(1, time_horizon)
            # TODO: should constrain sampling based on t... actuation etc.
            config = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            state = np.append(config, t)
            if self.world.is_collision_free_state(state):
                return state

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
