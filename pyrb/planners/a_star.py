import bisect
import numpy as np
import math

from collections import defaultdict


class QueueElement:

    def __init__(self, cost, state):
        self.cost = cost
        self.state = state

    def __lt__(self, other):
        return self.cost < other.cost


class AstarPlanner:

    def __init__(self, discrete_world):
        self.discrete_world = discrete_world
        self.radius_goal_region = 1e-3
        self.g = defaultdict(lambda: math.inf)

    def plan(self, state_start, state_goal):
        est_cost_to_goal = self.discrete_world.estimate_cost_to_goal(state_start, state_goal)
        queue = [QueueElement(est_cost_to_goal, state_start)]
        closed = set()
        actions = self.discrete_world.get_discrete_actions()
        self.g.clear()
        g = self.g
        g[tuple(state_start)] = 0
        path = []
        parents = {}
        while queue:
            element = queue.pop(0)
            state = element.state
            key_state = tuple(state)
            cost_to_start = g[key_state]
            closed.add(key_state)
            if self.discrete_world.is_in_goal_region(state, state_goal):
                path = self.extract_path(parents, state_start, state)
                break
            for a in actions:
                state_neigh = state + a
                key_state_neigh = tuple(state_neigh)
                if key_state_neigh in closed:
                    continue
                if not self.discrete_world.is_state_feasible(state):
                    continue
                if not self.discrete_world.is_collision_free_transition(state, state_neigh):
                    continue
                transition_cost = np.linalg.norm(a)
                cost_start_to_neighbor = cost_to_start + transition_cost
                if cost_start_to_neighbor < g[key_state_neigh]:
                    g[key_state_neigh] = cost_start_to_neighbor
                    parents[key_state_neigh] = key_state
                    est_cost_left = self.discrete_world.estimate_cost_to_goal(state_neigh, state_goal)
                    est_total_cost = cost_start_to_neighbor + est_cost_left
                    bisect.insort_left(queue, QueueElement(est_total_cost, state_neigh))
        return path

    @staticmethod
    def extract_path(parents, state_start, state_goal):
        key_start = tuple(state_start)
        key_goal = tuple(state_goal)
        path = [key_goal]
        key = key_goal
        while key != key_start:
            key = parents[key]
            path.append(key)
        path.reverse()
        return path
