import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


from pyrb.mp.utils.constants import LocalPlannerStatus
from pyrb.coll_det.simple import collision_detect_ball_and_line_segment
from pyrb.mp.planners.post_processing import post_process_path_continuous



class BallObstacle:

    def __init__(self):
        self.r = 1
        self.pos = np.array([5, 5])


class DummyLocalPlanner:

    def __init__(self, ball_obstacle):
        self.ball_r = ball_obstacle.r
        self.ball_pos = ball_obstacle.pos

    def plan(
            self,
            space,
            state_src,
            state_dst,
            goal_region,
            max_distance=np.inf
    ):
        collision = collision_detect_ball_and_line_segment(self.ball_r, self.ball_pos, state_src, state_dst)
        status = LocalPlannerStatus.TRAPPED if collision else LocalPlannerStatus.REACHED
        return status, state_dst


obstacle = BallObstacle()
local_planner = DummyLocalPlanner(obstacle)
path = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 0.0],
    [8.0, 0.0],
    [10.0, 10.0]
])

# i_segment, j_segment, x, y = sample_line_uniformly_on_path(path)
# line = np.vstack([x, y])
# path_pp = np.vstack(
#     [path_pp[:(i_segment+1), :], x.reshape(1, -1), y.reshape(1, -1), path_pp[j_segment+1:, :]]
# )

path_pp = post_process_path_continuous(
    None,
    local_planner,
    path,
    goal_region=None,
    max_cnt=100
)

fig, ax = plt.subplots(1, 1)
ball_r = 1
ball_pos = np.array([5, 5])
ax.add_patch(Circle(tuple(local_planner.ball_pos), local_planner.ball_r))
ax.plot(path[:, 0], path[:, 1], marker=".")

# ax.plot(line[:, 0], line[:, 1], marker=".", lw=4)
ax.plot(path_pp[:, 0], path_pp[:, 1], marker=".")

# ax.plot(path_pp[:, 0], path_pp[:, 1], marker=".")
ax.set_aspect("equal")
plt.show()
