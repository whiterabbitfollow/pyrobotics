import numpy as np


def path_distance(p):
    return np.linalg.norm(p[1:, :] - p[:-1, :], axis=1).sum()

