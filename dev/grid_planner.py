import pickle
from collections import defaultdict
import math
from typing import List, Tuple, Any

import numpy as np

import time
import tqdm


class StaticGridPlanner:

    def __init__(self, env):
        self.env = env
        self.agent = env.agent
        self.edges = defaultdict(lambda: [])
        self.verts = None
        self.gradients = None
        self.g = defaultdict(lambda: math.inf)
        self.distances = {}

    def create_graph(self):
        self.q_goal = None
        self.gradients = None

        map(lambda x: x.clear(), (self.edges, self.verts, self.distances))
        self.verts = np.array([])
        self.init_collision_free_vertices()
        self.init_collision_free_edges()

    def init_collision_free_vertices(self):
        grids = []
        for joint_nr in range(self.agent.nr_joints):
            grid_joint = np.arange(
                self.agent.joint_limits[joint_nr, 0],
                self.agent.joint_limits[joint_nr, 1],
                self.agent.max_actuation
            )
            grids.append(grid_joint)
        points = np.meshgrid(*grids)
        verts = np.c_[[p.ravel() for p in points]].T
        coll_free_configs = []
        for q_arr in verts:
            is_collision_free_state = self.is_state_collision_free(q_arr)
            if is_collision_free_state:
                coll_free_configs.append(q_arr)
                self.distances[tuple(q_arr)] = self.env.get_min_collision_distance()
        self.verts = np.array(coll_free_configs)

    def is_state_collision_free(self, q_arr):
        self.agent.set_config(q_arr)
        return not self.env.is_collision()

    def init_collision_free_edges(self):
        for q_arr in self.verts:
            self.create_edges_from_state(q_arr)

    def create_edges_from_state(self, q_arr: np.array):
        collision_free_neighs = self.get_collision_free_neighbors(q_arr)
        self.edges[tuple(q_arr)].extend(collision_free_neighs)

    def get_collision_free_neighbors(self, q_arr: np.array):
        neighbors = self.get_closest_neighbors(q_arr)
        collision_free_neigh = [
            tuple(q_neigh)
            for q_neigh in neighbors
            if self.is_edge_free(q_arr, q_neigh)
        ]
        return collision_free_neigh

    def get_closest_neighbors(self, q) -> np.ndarray:
        radius = self.agent.max_actuation
        dists = np.linalg.norm(q - self.verts, axis=1, ord=np.inf)
        neighs = self.verts[(dists <= radius * 1.01) & (dists > 1e-5)]
        return neighs

    def is_edge_free(self, q_arr_src, q_arr_dst):
        is_collision, _ = self.env.collision_check_transition(q_arr_src, q_arr_dst)
        return not is_collision

    def set_goal(self, q_goal):
        q_goal = tuple(q_goal)
        if not self.is_state_in_graph(q_goal):
            assert self.is_state_collision_free(q_goal), f"state {q_goal} not collision free"
            self.add_new_state_to_graph(q_goal)
        self.q_goal = q_goal
        self.env.set_goal_state(self.q_goal)
        self.g.clear()
        self.propagate_distances_from_goal()

    def is_state_in_graph(self, q):
        return (np.linalg.norm(self.verts - q) < 1e-6).any()

    def compute_collision_distance(self, q_arr):
        self.agent.set_config(q_arr)
        return self.env.get_min_collision_distance()

    def add_new_state_to_graph(self, q):
        q_arr = np.array(q)
        self.verts = np.vstack([self.verts, q_arr])
        self.create_edges_from_state(q_arr)
        for q_neigh in self.edges[q]:
            self.edges[q_neigh].append(q)
        self.distances[q] = self.compute_collision_distance(q_arr)

    def propagate_distances_from_goal(self):
        queue = []
        self.g.clear()
        g = self.g
        q_goal = self.q_goal
        g[q_goal] = 0
        queue.append(self.q_goal)
        # TODO: add goal to graph before....
        while queue:
            q = queue.pop(0)
            neigh = self.edges[q]
            for q_neigh in neigh:
                actuator_cost = self.compute_distance(q, q_neigh)
                cost = actuator_cost
                distance = max(self.distances[q_neigh], 1e-6)
                cost += (0.01 / distance) * 0.1
                if (g[q] + cost) < g[q_neigh]:
                    g[q_neigh] = g[q] + cost
                    queue.append(q_neigh)

    def solve_query(self, q_init):
        path = [q_init]
        q = q_init
        while self.compute_distance(q, self.q_goal) > 1e-3:
            q = self.get_best_next_state(q)
            path.append(q)
        return path

    def get_best_next_state(self, q) -> Tuple:
        neigh = self.edges[q]
        if not neigh:
            raise RuntimeError(f"No neighs exists for {q}")
        best_neigh = min(neigh, key=lambda x: self.g[x])
        best_cost = self.g[best_neigh]
        if best_cost == math.inf:
            raise RuntimeError(f"Cannot find a collision free path for {q}")
        return best_neigh

    def compute_gradients(self):
        gradients = []
        for q_arr in self.verts:
            q_nxt = self.get_best_next_state(tuple(q_arr))
            q_nxt_arr = np.array(q_nxt)
            action = q_nxt_arr - q_arr
            gradients.append(action)
        self.gradients = np.array(gradients)

    @staticmethod
    def compute_distance(q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def interpolate_action_from_neighbors(self, q):
        radius = self.agent.max_actuation
        dists_all = np.linalg.norm(q - self.verts, axis=1, ord=np.inf)
        mask = (dists_all <= radius * 1.01)
        neighs = self.verts[mask]
        wghts = 1/(np.linalg.norm(q - neighs, axis=1) + 1e-11)
        action = ((self.gradients[mask] * wghts.reshape(-1, 1))/wghts.sum()).sum(axis=0)
        return action





if __name__ == "__main__":
    from envs.robot_robot_collab import RobAndRobCollabGoalEnv
    from envs.utils import vizualize_path, plot_path, get_collision_boundary

    env_kwargs = {
        "static_adversary": True,
        "adversary_kwargs": {"static_pose": np.array([2.98925226, 1.3335978])},
        "reset_agent_mode": "random"
    }
    env = RobAndRobCollabGoalEnv(**env_kwargs)
    env.reset()


    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    collision_boundary = get_collision_boundary(env)

    planner = StaticGridPlanner(env)
    import time
    time_s = time.time()
    planner.create_graph()

    q_goal = planner.verts[100]
    planner.set_goal(tuple(q_goal))
    q_start = planner.verts[399].copy()
    path_expert = np.array(planner.solve_query(tuple(q_start)))
    planner.compute_gradients()
    path = [q_start.copy()]
    max_steps = 300
    step_nr = 0
    q_arr = q_start
    while np.linalg.norm(q_arr-q_goal) > 0.1 and step_nr < max_steps:
        action = planner.interpolate_action_from_neighbors(q_arr)
        q_arr += action + np.random.random() * 0.15
        path.append(q_arr + action)
        step_nr += 1
    path = np.array(path)
    fig, ax = plt.subplots(1, 1)
    plt.fill_between(collision_boundary[:, 0], collision_boundary[:, 1], alpha=.3, color="red")
    ax.plot(path[:, 0], path[:, 1])
    ax.plot(path_expert[:, 0], path_expert[:, 1])
    ax.add_patch(Circle(q_goal, radius=0.1, alpha=.1, color="red"))
    plt.show()
    # cost = [planner.g[tuple(vert)] for vert in planner.verts]
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(planner.verts[:, 0], planner.verts[:, 1], cost, c=cost)
    # plt.show()

    # print(f"create_graph: time elapsed {time.time() - time_s}")
    # time_s = time.time()
    # print(f"set_goal: time elapsed {time.time() - time_s}")
    # print(f"nr verts: {planner.verts.shape}")
    # paths = []
    import tqdm
    # tbar = tqdm.tqdm(total=planner.verts.shape[0] ** 2)
    # for q in planner.verts:
    #     planner.set_goal(tuple(q))
    #     for qq in planner.verts:
    #         if (qq == q).all():
    #             continue
    #         path = planner.solve_query(tuple(qq))
    #         print(len(path))
    #         paths.append(path)
    #     break
    #     # print(f"Path: {path}")
    #     # path = np.array(path)
    #     # vizualize_path(env, path, render_kwargs={"matplotlib_pause": 0.1})
    #     # plot_path(env, path)