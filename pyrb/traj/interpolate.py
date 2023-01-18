import numpy as np


def interpolate_path(sparse_path, segment_interpolator):
    path = []
    for src, dst in zip(sparse_path[:-1], sparse_path[1:]):
        segment = segment_interpolator(src, dst)
        segment_no_dst = segment[:-1, :]
        path.append(segment_no_dst)
    return np.vstack(path + [sparse_path[-1]])



def interpolate_along_line_l_infty_dim_2(config_src, config_dst, max_distance_per_step_per_dim):
    max_distance_per_step = compute_max_distance_l_infty_dim_2(config_src, config_dst, max_distance_per_step_per_dim)
    return interpolate_along_line(config_src, config_dst, max_distance_per_step)


def compute_max_distance_l_infty_dim_2(config_src, config_dst, dim_max_distance):
    w, h = np.abs((config_dst - config_src).ravel())
    angle = np.arctan2(h, w)
    if w > h:
        distance_along_line = dim_max_distance / np.cos(angle)
    else:
        distance_along_line = dim_max_distance / np.sin(angle)
    return distance_along_line


def interpolate_along_line(config_src, config_dst, max_distance_per_step):
    distance = np.linalg.norm(config_dst - config_src)
    betas = np.arange(0, distance, max_distance_per_step) / distance
    if not np.isclose(betas[-1], 1):
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

