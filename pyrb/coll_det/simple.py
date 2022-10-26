import numpy as np


def collision_detect_ball_and_line_segment(ball_r, ball_pos, x_1, x_2):
    vec_line_seg = (x_2 - x_1).reshape(-1, 1)
    n = compute_normal(u=(ball_pos - x_1).reshape(-1, 1), v=vec_line_seg)
    smallest_distance_to_line = np.linalg.norm(n)
    line_intersects_ball = smallest_distance_to_line < ball_r
    if line_intersects_ball:
        point_on_line = ball_pos - n.ravel()
        direction = (point_on_line - x_1).reshape(-1, 1)
        dist_to_point = np.linalg.norm(direction)
        dist_len_seg = np.linalg.norm(vec_line_seg)
        cos_theta = ((vec_line_seg.T @ direction) / (dist_len_seg * dist_to_point)).ravel()
        is_point_on_line_behind_x_1 = cos_theta < 0
        if is_point_on_line_behind_x_1:
            collision = is_point_within_ball(x_1, ball_pos, ball_r)
        else:
            point_is_on_line_segment = dist_to_point <= dist_len_seg
            if not point_is_on_line_segment:
                collision = end_points_within_ball(x_1, x_2, ball_pos, ball_r)
            else:
                collision = True
    else:
        collision = False
    return collision


def compute_normal(u, v):
    return u - (u.T @ v) / (v.T @ v) * v


def end_points_within_ball(x_1, x_2, ball_pos, ball_r):
    return is_point_within_ball(x_1, ball_pos, ball_r) \
           or \
           is_point_within_ball(x_2, ball_pos, ball_r)


def is_point_within_ball(x, ball_pos, ball_r):
    return np.linalg.norm(x - ball_pos) < ball_r
