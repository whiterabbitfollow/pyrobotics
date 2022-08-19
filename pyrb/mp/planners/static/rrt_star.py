import logging
import time

import numpy as np

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.utils.tree import TreeRewire
from pyrb.mp.utils.utils import start_timer, is_vertex_in_goal_region, compile_planning_data
from pyrb.mp.planners.static.local_planners import LocalPlanner


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStarPlanner:

    def __init__(
            self,
            world: BaseMPWorld,
            max_nr_vertices=int(1e4),
            max_distance_local_planner=0.5,
            min_step_size_local_planner=0.01,
            nearest_radius=.2
    ):
        self.state_goal = None
        self.vert_cnt = 0
        self.goal_region_radius = 1e-1
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=min_step_size_local_planner,
            max_distance=max_distance_local_planner,
            global_goal_region_radius=self.goal_region_radius
        )
        self.tree = TreeRewire(
            max_nr_vertices=max_nr_vertices,
            vertex_dim=world.robot.nr_joints,
            nearest_radius=nearest_radius,
            local_planner=self.local_planner
        )

    def clear(self):
        self.tree.clear()

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.clear()
        self.state_goal = state_goal
        self.tree.add_vertex(state_start)
        path = np.array([]).reshape((0, state_start.size))
        time_s, time_elapsed = start_timer()
        while not self.tree.is_full() and time_elapsed < max_planning_time and path.size == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, self.state_goal)
            state_new = local_path[-1] if local_path.size > 0 else None
            if state_new is not None:
                self.tree.rewire_nearest(i_nearest, state_new)
                if is_vertex_in_goal_region(state_new, state_goal, self.goal_region_radius):
                    logger.debug("Found path to goal!!!")
                    path = self.find_path(state_start)
            time_elapsed = time.time() - time_s
        return path, compile_planning_data(path, time_elapsed, self.tree.vert_cnt)

    def find_path(self, state_start):
        vertices = self.tree.get_vertices()
        distances = np.linalg.norm(self.state_goal - vertices, axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            indices = mask_vertices_goal.nonzero()[0]
            i_min_cost = indices[np.argmin(self.tree.cost_to_verts[indices])]
            path = self.tree.find_path_to_root_from_vertex_index(i_min_cost)
            path = path[::-1]
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state


class RRTStarPlannerModified(RRTStarPlanner):

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.clear()
        self.state_goal = state_goal
        self.tree.add_vertex(state_start)
        path = np.array([]).reshape((-1, ) + state_goal.shape)
        time_s, time_elapsed = start_timer()
        while not self.tree.is_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, state_goal)
            for state_new in local_path:
                self.tree.rewire_nearest(i_nearest, state_new)
                if is_vertex_in_goal_region(state_new, state_goal, self.goal_region_radius):
                    logger.debug("Found path to goal!!!")
                    path = self.find_path(state_start)
                    break
                i_nearest = self.vert_cnt - 1
            time_elapsed = time.time() - time_s
        return path, compile_planning_data(path, time_elapsed, self.tree.vert_cnt)

