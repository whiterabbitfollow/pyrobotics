import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def intersection_sphere_line_segment(sphere_radius, sphere_center, line_seg_start, line_seg_end):
    v_1 = line_seg_start.ravel()
    v_2 = line_seg_end.ravel()

    v_dir = v_2 - v_1
    v_mid = (v_1 + v_2)/2

    v_2_r = v_2 - v_mid
    v_1_r = v_1 - v_mid

    v_dir_unit = v_dir / np.linalg.norm(v_dir)
    u = sphere_center - v_mid

    magnitude = (u.T @ v_dir_unit / (v_dir_unit.T @ v_dir_unit))
    v_proj = magnitude * v_dir_unit
    u_n = u - v_proj

    smallest_dist = np.linalg.norm(u_n)

    v_2_k = (v_dir_unit/v_2_r)[0]
    v_1_k = (v_dir_unit/v_1_r)[0]

    v_2_dist = np.linalg.norm(v_2_r - u)
    v_1_dist = np.linalg.norm(v_1_r - u)

    if v_1_k <= magnitude <= v_2_k:
        intersect = smallest_dist < sphere_radius
    elif magnitude < v_1_k:
        intersect = v_1_dist < sphere_radius
    else:
        intersect = v_2_dist < sphere_radius
    return intersect


fig, ax = plt.subplots(1, 1)
ax.add_patch(Circle(tuple(u), radius=radius, alpha=.1))
ax.scatter(*tuple(u))
ax.plot(vs[:, 0], vs[:, 1])
ax.set_aspect("equal")
plt.show()



