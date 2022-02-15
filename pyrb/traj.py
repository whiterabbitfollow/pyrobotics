import numpy as np


def generate_random_trajectory(
        joint_limits,
        nr_via_points=10,
        delta_t=0.1,
        nr_points=500,
        start_pos=None
):
    nr_joints = joint_limits.shape[0]
    beta = np.hstack([np.random.uniform(l, u, (nr_via_points,)).reshape(-1, 1) for l, u in joint_limits])
    if start_pos is not None:
        beta[0, :] = start_pos
    beta_d = np.random.uniform(0, 2 * np.pi, (nr_via_points, nr_joints))
    coeffs = []
    for i in range(nr_via_points):
        i_nxt = (i + 1) % nr_via_points
        a_0 = beta[i]
        a_1 = beta_d[i]
        a_2 = 3 * beta[i_nxt] - 3 * beta[i] - 2 * beta_d[i] * delta_t - beta_d[i_nxt] * delta_t
        a_2 /= (delta_t ** 2)
        a_3 = 2 * beta[i] + (beta_d[i] + beta_d[i_nxt]) * delta_t - 2 * beta[i_nxt]
        a_3 /= (delta_t ** 3)
        coeffs.append((a_0, a_1, a_2, a_3))
    return calculate_psaj(coeffs, nr_via_points, delta_t, nr_points=nr_points)


def calculate_psaj(coeffs, nr_viapoints, delta_t, nr_points=500):
    N = nr_viapoints
    ts = np.linspace(0, N * delta_t, nr_points)
    poses, spds, accs, jrks = [], [], [], []
    for t in ts:
        t %= (delta_t * N)
        t_trunc = int(t // delta_t)
        a_0, a_1, a_2, a_3 = coeffs[t_trunc]
        dt = t - (t_trunc * delta_t)
        pos = a_0 + a_1 * dt    + a_2 * dt ** 2 + a_3 * dt ** 3
        spd =       a_1         + 2 * a_2 * dt  + 3 * a_3 * dt ** 2
        acc =                     2 * a_2       + 6 * a_3 * dt
        jrk =                                     6 * a_3
        poses.append(pos)
        spds.append(spd)
        accs.append(acc)
        jrks.append(jrk)
    return np.c_[poses], np.c_[spds], np.c_[accs], np.c_[jrks]



def calculate_poses(coeffs, nr_viapoints, delta_t, nr_points=500):
    N = nr_viapoints
    ts = np.linspace(0, N * delta_t, nr_points)
    poses = []
    for t in ts:
        t %= (delta_t * N)
        t_trunc = int(t // delta_t)
        a_0, a_1, a_2, a_3 = coeffs[t_trunc]
        dt = t - (t_trunc * delta_t)
        pos = a_0 + a_1 * dt + a_2 * dt ** 2 + a_3 * dt ** 3
        poses.append(pos)
    return np.c_[poses]