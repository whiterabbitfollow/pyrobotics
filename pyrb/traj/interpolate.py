import numpy as np


def interpolate_path(sparse_path, segment_interpolator):
    path = []
    for config_src, config_dst in zip(sparse_path[:-1], sparse_path[1:]):
        segment = segment_interpolator(config_src, config_dst)
        segment_no_dst = segment[:-1, :]
        path.append(segment_no_dst)
    return np.vstack(path + [sparse_path[-1]])


def interpolate_along_line_l_infty_dim_2(config_src, config_dst, max_distance_per_step_per_dim):
    w, h = np.abs((config_dst - config_src).ravel())
    angle = np.arctan2(h, w)
    if w > h:
        max_distance_per_step = max_distance_per_step_per_dim / np.cos(angle)
    else:
        max_distance_per_step = max_distance_per_step_per_dim / np.sin(angle)
    return interpolate_along_line(config_src, config_dst, max_distance_per_step)


def interpolate_along_line(config_src, config_dst, max_distance_per_step):
    distance = np.linalg.norm(config_dst - config_src)
    betas = np.arange(0, distance, max_distance_per_step) / distance
    betas = np.append(betas, 1)
    betas = betas.reshape(-1, 1)
    nr_steps = betas.shape[0]
    config_src_many = np.tile(config_src.reshape(1, -1), (nr_steps, 1))
    config_dst_many = np.tile(config_dst.reshape(1, -1), (nr_steps, 1))
    configs = (1 - betas) * config_src_many + betas * config_dst_many
    return configs


def interpolate_single_point_along_line(config_src, config_dst, distance_to_point):
    distance_line = np.linalg.norm(config_dst - config_src)
    beta = distance_to_point / distance_line
    return (1 - beta) * config_src + beta * config_dst

