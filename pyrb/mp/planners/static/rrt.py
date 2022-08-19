import logging
import time

import numpy as np

from pyrb.mp.base_world import BaseMPWorld
from pyrb.mp.planners.moving.rrt import LocalPlanner
from pyrb.mp.utils.utils import PlanningData, Status, start_timer, is_vertex_in_goal_region
from pyrb.mp.utils.tree import Tree

logger = logging.getLogger()


class RRTPlanner:

    def __init__(self, world: BaseMPWorld, max_nr_vertices=int(1e4), max_distance_local_planner=0.5):
        self.tree = Tree(max_nr_vertices=max_nr_vertices, vertex_dim=world.robot.nr_joints)
        self.state_goal = None
        self.max_distance_local_planner = max_distance_local_planner
        self.goal_region_radius = .1
        self.world = world
        self.configuration_limits = self.world.robot.get_joint_limits()
        self.local_planner = LocalPlanner(
            self.world,
            min_step_size=0.01,
            max_distance=max_distance_local_planner,
            global_goal_region_radius=self.goal_region_radius
        )

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.tree.clear()
        self.state_goal = state_goal
        self.tree.add_vertex(state_start)
        path = []
        time_s, time_elapsed = start_timer()
        while not self.tree.is_full() and time_elapsed < max_planning_time and len(path) == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, state_goal)
            state_new = local_path[-1] if local_path.size > 0 else None
            if state_new is not None:
                self.tree.append_vertex(state_new, i_parent=i_nearest)
                if is_vertex_in_goal_region(state_new, self.state_goal, self.goal_region_radius):
                    logger.debug("Found vertex in goal region!")
                    path = self.find_path(state_start)
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)


    def find_path(self, state_start):
        vertices = self.tree.get_vertices()
        distances = np.linalg.norm(self.state_goal - vertices, axis=1)
        mask_vertices_goal = distances < self.goal_region_radius
        if mask_vertices_goal.any():
            i = mask_vertices_goal.nonzero()[0][0]
            state = vertices[i, :]
            path = [state]
            while (state != state_start).any():
                i = self.tree.get_vertex_parent_index(i)
                state = vertices[i, :]
                path.append(state)
            path.reverse()
            path = np.vstack(path)
        else:
            path = np.array([]).reshape((-1,) + state_start.shape)
        return path

    def sample_collision_free_config(self):
        while True:
            state = np.random.uniform(self.configuration_limits[:, 0], self.configuration_limits[:, 1])
            if self.world.is_collision_free_state(state):
                return state

    def compile_planning_data(self, path, time_elapsed):
        status = Status.SUCCESS if path.size else Status.FAILURE
        return PlanningData(status=status, time_taken=time_elapsed, nr_verts=self.tree.vert_cnt)


class RRTPlannerModified(RRTPlanner):

    def plan(self, state_start, state_goal, max_planning_time=np.inf):
        self.tree.add_vertex(state_start)
        self.state_goal = state_goal
        path = np.array([]).reshape((-1,) + state_start.shape)
        time_s, time_elapsed = start_timer()
        while not self.tree.is_full() and time_elapsed < max_planning_time and path.size == 0:
            state_free = self.sample_collision_free_config()
            i_nearest, state_nearest = self.tree.find_nearest_vertex(state_free)
            local_path = self.local_planner.plan(state_nearest, state_free, self.state_goal)
            i_prev = i_nearest
            for state in local_path:
                i_current = self.tree.vert_cnt
                self.tree.append_vertex(state, i_parent=i_prev)
                i_prev = i_current
                if is_vertex_in_goal_region(state, state_goal, self.goal_region_radius):
                    path = self.find_path(state_start)
                    break
            time_elapsed = time.time() - time_s
        return path, self.compile_planning_data(path, time_elapsed)
