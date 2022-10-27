import matplotlib.pyplot as plt
import numpy as np

from pyrb.mp.planners.local_planners import LocalPlannerSpaceTime
from pyrb.mp.utils.goal_regions import RealVectorMinimizingTimeGoalRegion


class DummySpaceTimeSpace:

    def __init__(self):
        self.max_time = 60

    def is_collision_free_transition(self, *args, **kwargs):
        return True


local_planner = LocalPlannerSpaceTime(
    max_actuation=1, max_distance=.1, min_coll_step_size=2
)

space = DummySpaceTimeSpace()

state_src = np.array([0, 0])
state_dst = np.array([1, 7])



goal_region = RealVectorMinimizingTimeGoalRegion(radius=.01)
goal_region.set_goal_state(np.array([1.1, np.inf]))


status, path = local_planner.bang_bang_control_to_destination(space, state_src, state_dst, max_distance=np.inf)
print(path)

plt.figure(1)
plt.plot(path[:, 0], path[:, 1], ls="--", marker=".")
plt.show()
