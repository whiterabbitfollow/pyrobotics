import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from pyrb.coll_det.simple import collision_detect_ball_and_line_segment

fig, ax = plt.subplots(1, 1)
ball_r = 1
ball_pos = np.array([5, 5])
ax.add_patch(Circle(tuple(ball_pos), ball_r))

for x1, x2 in [
    ([5.6, 5.6], [5.5, 5.5]),
    ([0, 0], [4.1, 4.1]),
    ([6, 6], [7, 10]),
    ([10, 0], [0, 10])
]:
    x1 = np.array(x1)
    x2 = np.array(x2)
    line_seg = np.vstack([x1, x2])
    collision = collision_detect_ball_and_line_segment(ball_r, ball_pos, x1, x2)
    collision_status = "red" if collision else "green"
    ax.plot(line_seg[:, 0], line_seg[:, 1], c=collision_status)

ax.set_aspect("equal")
plt.show()

