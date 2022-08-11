import pickle
import time
from collections import defaultdict
import math

from typing import List, Tuple
import numpy as np
import tqdm


class DynamicGridPlanner:

    def __init__(self, env, max_nr_steps=60, verbose=False, include_distance=True):
        self.env = env
        self.nr_max_steps = max_nr_steps
        self.edges_bkw = defaultdict(lambda: [])
        self.edges = defaultdict(lambda: [])
        self.verts: List[np.ndarray] = []
        self.g = defaultdict(lambda: math.inf)
        self.distance = {}
        self.include_distance = include_distance
        self.q_goal = None
        self.verbose = verbose

    def create_graph(self):
        self.q_goal = None
        map(lambda x: x.clear(), (self.edges_bkw, self.edges, self.verts, self.g))
        self.add_collision_free_vertices(self.nr_max_steps)
        self.add_collision_free_edges()

    def add_collision_free_vertices(self, nr_times_to_add):
        t_start = len(self.verts)
        time_steps = range(t_start, t_start + nr_times_to_add)
        if self.verbose:
            time_steps = tqdm.tqdm(time_steps)
        for t in time_steps:
            if self.verbose:
                time_steps.set_description("Sampling coll.free vertices")
            verts, distances = self.env.grid_collision_free_vertices(t, return_distance=True)
            self.verts.append(np.array(verts))
            self.distance.update(distances)
        self.nr_max_steps = len(self.verts)

    def add_collision_free_edges(self, t_start=1, t_end=None):
        t_end = self.nr_max_steps if t_end is None else t_end
        times = range(t_start, t_end)[::-1]
        nr_verts_all_times = sum((self.verts[t].shape[0] for t in times))
        if self.verbose:
            pbar = tqdm.tqdm(total=nr_verts_all_times)
        for t in times:
            if self.verbose:
                pbar.set_description(f"Processing time {t}, nr verts: {len(self.verts[t])}")
            for q in self.verts[t]:
                self.create_edges_from_state_time(q, t)
                if self.verbose:
                    pbar.update(1)

    def create_edges_from_state_time(self, q: np.ndarray, t: float):
        collision_free_neighs = self.get_collision_free_neighbors_backward(q, t)
        if not collision_free_neighs:
            # TODO: add collision free neighs?
            pass
        # collision free transitions from q_neigh, t-1 to q, t
        # all states that can transition to q from prev time step
        assert (tuple(q), t) not in self.edges_bkw
        self.edges_bkw[(tuple(q), t)].extend(collision_free_neighs)
        # collision free transitions from q_neigh, t-1 to q. t
        for q_neigh in collision_free_neighs:
            # all states I can transition to
            self.edges[(q_neigh, t - 1)].append(tuple(q))

    def get_collision_free_neighbors_backward(self, q, t):
        """
        Get collision free neighbors:
         (q_neigh, t - 1) - > (q, t)
        """
        neighbors = self.get_closest_neighbors(q, t - 1)
        collision_free_neigh = [tuple(q_neigh) for q_neigh in neighbors if self.env.is_edge_free(t - 1, q_neigh, q)]
        return collision_free_neigh

    def get_collision_free_neighbors_forward(self, q, t):
        neighbors = self.get_closest_neighbors(q, t + 1)
        collision_free_neigh = [tuple(q_neigh) for q_neigh in neighbors if self.env.is_edge_free(t, q, q_neigh)]
        return collision_free_neigh

    def get_closest_neighbors(self, q, t) -> np.ndarray:
        # TODO: problem
        # project action space to next state, get states within that ball.
        radius = self.env.agent.max_actuation
        dists = np.linalg.norm(q - self.verts[t], axis=1, ord=np.inf)
        neighs = self.verts[t][dists <= radius * 1.01]
        return neighs

    def set_goal(self, goal):
        self.q_goal = goal
        self.g.clear()
        self.propagate_distances_from_goal()

    def propagate_distances_from_goal(self):
        queue = []
        self.g.clear()
        g = self.g
        q_goal = self.q_goal
        max_time_step = len(self.verts)
        for t in range(1, max_time_step)[::-1]:
            if not self.env.is_state_collision_free(t, q_goal):
                continue
            if (q_goal, t) not in self.edges_bkw:
                self.create_edges_from_state_time(np.array(q_goal), t)
            # if no neighbours?
            g[(q_goal, t)] = 0
            queue.append((q_goal, t))
        while queue:
            # going backwards in time...
            q, t = queue.pop(0)
            # neigh that can be transitioned t-1, collision free (q_neigh, t-1) -> (q, t)
            neigh = self.edges_bkw[(q, t)]
            for q_neigh in neigh:
                actuator_cost = self.compute_distance(q, q_neigh)
                cost = actuator_cost
                if self.distance and self.include_distance:
                    distance = max(self.distance[(q_neigh, t-1)], 1e-6)
                    cost += (0.01/distance) * 0.1
                # g[(q, t)] + cost -> cost to transition to goal, from q_neigh through q (backwards in time).
                # q[(q_neigh, t-1)] -> current cost from q_neigh at t-1, to goal
                if t > 0 and (g[(q, t)] + cost) < g[(q_neigh, t-1)]:
                    # shorter to transition through q when going from q_neigh to goal
                    g[(q_neigh, t-1)] = g[(q, t)] + cost
                    # TODO: if t > 1
                    queue.append((q_neigh, t-1))

    @staticmethod
    def compute_distance(q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def solve_query(self, q_init: Tuple, t=0):
        path = [(q_init, t)]
        q = q_init
        while self.compute_distance(q, self.q_goal) > 1e-3 and (t + 1) < self.nr_max_steps:
            q = self.get_best_next_state(q, t)
            path.append((q, t + 1))
            t += 1
        return path

    def add_state_to_graph_and_all_times(self, q_arr: np.ndarray, t_init):
        if not self.env.is_state_collision_free(t_init, q_arr):
            raise RuntimeError(f"Init state {q_arr} is not collision free time {t_init}")
        for t in range(0, len(self.verts)):
            if self.env.is_state_collision_free(t, q_arr):
                self.verts[t] = np.vstack([self.verts[t], q_arr])
        for t in range(1, len(self.verts))[::-1]:
            self.create_edges_from_state_time(q_arr, t)
        t = 0
        collision_free_neighs = self.get_collision_free_neighbors_forward(q_arr, t)
        self.edges[(tuple(q_arr), t)].extend(collision_free_neighs)
        self.propagate_distances_from_goal()

    def add_state_to_graph(self, q_arr: np.ndarray, t, propagate=False):
        if not self.env.is_state_collision_free(t, q_arr):
            raise RuntimeError("Init state is not collision free")
        self.verts[t] = np.vstack([self.verts[t], q_arr])
        self.create_edges_from_state_time(q_arr, t)
        if propagate:
            self.propagate_distances_from_goal()

    def get_best_next_state(self, q, t) -> Tuple:
        neigh = self.edges[(q, t)]
        if not neigh:
            raise RuntimeError("No neighs for %s at time %s (t+1)" % (q, t+1))
        best_neigh = min(neigh, key=lambda x: self.g[(x, t + 1)])
        best_cost = self.g[(best_neigh, t + 1)]
        if best_cost == math.inf:
            raise RuntimeError("Cannot find a collision free path for %s at time %s (t+1)" % (q, t+1))
        return best_neigh

    def save_state_to_pickle(self, pickle_file_path):
        env_state = self.env.get_system_params()    # self.env.get_state
        with open(pickle_file_path, "wb") as fp:
            edges_bkw = {**self.edges_bkw}
            edges = {**self.edges}
            pickle.dump({
                "edges_bkw": edges_bkw,
                "edges": edges,
                "verts": self.verts,
                "distance": self.distance,
                "env_state": env_state
            },
                fp
            )

    def load_state_from_pickle(self, pickle_file_path):
        self.env.clear()
        with open(pickle_file_path, "rb") as fp:
            state_dict = pickle.load(fp)
            self.edges_bkw.clear()
            self.edges.clear()
            self.edges_bkw.update(state_dict["edges_bkw"])
            self.edges.update(state_dict["edges"])
            self.distance.update(state_dict["distance"])
            self.verts = state_dict["verts"]
            self.nr_max_steps = len(self.verts)
            self.env.set_system_params(**state_dict["env_state"])


def show_planned_path(mpenv, env, path):
    env.adversary.traj = mpenv.adversary.traj
    q_start, t_start = path[0]
    env.adversary.set_time(t_start)

    q_goal, t_end = path[-1]
    env.agent.set_config(q_start)
    env.set_goal_state(q_goal)
    for (q_nxt, _), (q, _) in zip(path[1:], path[:-1]):
        action = np.array(q_nxt) - np.array(q)
        obs, reward, done, info = env.step(action)
        env.render(mode="matplotlib", matplotlib_pause=0.1)
    return


def visualize_verts_all_times(planner, q_start=None, q_goal=None):
    import matplotlib.pyplot as plt
    plt.figure(1)
    for t in range(len(planner.verts)):
        visualize_verts_for_time(planner, q_start, q_goal, t, show=False)
        plt.pause(0.1)
        plt.cla()


def visualize_verts_for_time(planner, q_start=None, q_goal=None, t=0, show=True):
    import matplotlib.pyplot as plt
    plt.scatter(planner.verts[t][:, 0], planner.verts[t][:, 1])
    qts = np.array([q for ((q, tt), cost) in planner.g.items() if cost < math.inf and tt == t])
    if qts.size > 0:
        plt.scatter(qts[:, 0], qts[:, 1], s=10)
    if q_start is not None:
        plt.scatter(q_start[0], q_start[1], color="black")
    if q_goal is not None:
        plt.scatter(q_goal[0], q_goal[1], color="red")
    plt.title(f"Time step: {t}")
    if show:
        plt.show()


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from planner.mp_time_envs import RobAndRobCollabMPEnv
    # from envs.env3d.adversary import Adversary3DOFManipulator
    # from envs.env3d.agent import Manipulator3DOF
    #
    # adversary = Adversary3DOFManipulator()
    # agent = Manipulator3DOF()
    from dev.mp_time_envs import RobAndRobCollabMPEnv
    from envs import make_2d_env_dynamic_grid

    env = make_2d_env_dynamic_grid(nr_lagging_steps=2, grid_size=64)
    mpenv = RobAndRobCollabMPEnv(env.agent, env.adversary)

    planner = DynamicGridPlanner(mpenv, verbose=True, include_distance=True, max_nr_steps=100)
    path_solved = r"/mnt/sdb2/phd/repos/dynenv/data/finite_horizon_planner/solved/train/solved_dplanner_th/solution_nr_0.pckl"
    planner.load_state_from_pickle(path_solved)

    q_start_arr = planner.verts[0][0, :]
    q_end_arr = planner.verts[-1][175, :]

    q_start = tuple(q_start_arr)
    q_end = tuple(q_end_arr)

    print(f"Start in {q_start}")
    print(f"End in {q_end}")

    planner.set_goal(q_end)
    # visualize_verts_all_times(planner, q_start, q_end)

    path = planner.solve_query(q_start, t=100)
    env.set_goal_state(q_end)

    while True:
        for q, t in path:
            env.agent.set_config(q)
            env.adversary.set_time(t)
            env.render(mode="human", title=f"t: {t}")
        time.sleep(1)

    # env = make_2d_env_dynamic_grid(nr_lagging_steps=2, grid_size=64)
    # mpenv = RobAndRobCollabMPEnv(env.agent, env.adversary)
    # mpenv.reset()
    # planner = DynamicGridPlanner(mpenv, verbose=True, include_distance=True, max_nr_steps=300)
    # planner.load_state_from_pickle("tmp.pckl")
    #
    # q_start_arr = planner.verts[0][150, :]
    # q_end_arr = planner.verts[0][0, :]
    #
    # # visualize_verts_all_times(planner)
    # q_start = tuple(q_start_arr)
    # q_end = tuple(q_end_arr)
    # planner.set_goal(q_end)
    # path = planner.solve_query(q_start)
    #
    # # visualize_verts_for_time(planner, q_start, q_end)
    # # #
    # # visualize_verts_all_times(planner, q_start, q_end)
    #
    # for q, t in path:
    #     env.agent.set_config(q)
    #     env.adversary.set_time(t)
    #     env.render(mode="human")
    # env.close()

    # path = planner.solve_query(q_start)
    # print(path)
    # path_to_solved = r"/mnt/sdb2/phd/repos/dynenv/data/solved/train/solved_dplanner/solution_nr_90.pckl"
    # # print("create graph")
    # # planner.create_graph()
    # # planner.save_state_to_pickle("tmp.pckl")
    # planner.load_state_from_pickle(path_to_solved)
    #

    #
    # env.agent.set_config(q_start_arr)
    # env.set_goal_state(q_end_arr)
    # env.adversary.set_time(0)
    # acc_reward = 0
    # print(path[0])
    # for (q, _), (q_nxt, _) in zip(path[:-1], path[1:]):
    #     action = np.array(q_nxt) - np.array(q)
    #     obs, reward, done, info = env.step(action)
    #     acc_reward += reward
    #     env.render(mode="human", matplotlib_pause=0.1)

