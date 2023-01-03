import numpy as np

from pyrb.kin import rot_z_to_SO3, rot_y_to_SO3, rot_x_to_SO3


def compute_rotation_matrix_from_angles(alpha, beta, gamma):
    return rot_z_to_SO3(gamma) @ rot_y_to_SO3(beta) @ rot_x_to_SO3(alpha)


def compute_rot_matrices(speed):
    nr_data_points, dim = speed.shape
    direction_x = speed
    direction_x = (direction_x.T / np.linalg.norm(direction_x, axis=1)).T
    e2 = np.eye(dim)[:, 1].reshape(-1, 1)
    direction_y = e2.ravel() - (direction_x.T * (direction_x @ e2).ravel()).T
    direction_y = (direction_y.T / np.linalg.norm(direction_y, axis=1)).T
    direction_z = np.cross(direction_x, direction_y)
    Rs = np.concatenate([direction_x.T[None, ...], direction_y.T[None, ...], direction_z.T[None, ...]], axis=0)
    # TODO: fix that we need to do transpose..
    return Rs

