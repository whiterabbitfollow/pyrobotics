import matplotlib.pyplot as plt
import numpy as np

from pyrb.mp.planners.local_planners import LocalPlannerSpaceTime
from pyrb.mp.planners.post_processing import post_process_path_space_time_systematic_minimal_time
from pyrb.mp.utils.goal_regions import RealVectorMinimizingTimeGoalRegion




class DummySpaceTimeSpace:

    def __init__(self):
        self.max_time = 60

    def is_collision_free_transition(self, state_src, state_dst, min_step_size):
        return True


local_planner = LocalPlannerSpaceTime(
    max_actuation=1, max_distance=.1, min_coll_step_size=2
)

space = DummySpaceTimeSpace()
path = np.array([
    [0, 0],
    [-1, 1],
    [-2, 2],
    [-3, 3],
    [2, 4],
    [1, 5],
    [0, 6],
    [1, 7],
])


goal_region = RealVectorMinimizingTimeGoalRegion(radius=.01)
goal_region.set_goal_state(np.array([1.1, np.inf]))


path_pp = post_process_path_space_time_systematic_minimal_time(space, local_planner, goal_region, path)

plt.figure(1)
plt.plot(path[:, 0], path[:, 1], ls="--", marker=".")
plt.plot(path_pp[:, 0], path_pp[:, 1], ls="--", marker=".")
plt.show()
