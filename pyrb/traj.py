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
    poly_coefficients = generate_cubic_coefficients_from_via_points_velocities(beta, beta_d, delta_t=delta_t)
    return calculate_psaj(poly_coefficients, delta_t=delta_t, nr_points=nr_points)


def generate_cubic_coefficients_from_via_points_velocities(points, velocities, delta_t):
    beta = points
    beta_d = velocities
    coeffs = []
    nr_via_points = beta.shape[0]
    for i in range(nr_via_points):
        i_nxt = (i + 1) % nr_via_points
        coeffs.append(calculate_cubic_coefficients_between_via_points(
            beta[i], beta_d[i], beta[i_nxt], beta_d[i_nxt], delta_t
        ))
    return coeffs


def calculate_cubic_coefficients_between_via_points(
        position_start,
        velocity_start,
        position_end,
        velocity_end,
        delta_t
    ):
    beta_s = position_start
    beta_d_s = velocity_start
    beta_e = position_end
    beta_d_e = velocity_end
    a_0 = beta_s
    a_1 = beta_d_s
    a_2 = 3 * beta_e - 3 * beta_s - 2 * beta_d_s * delta_t - beta_d_e * delta_t
    a_2 /= (delta_t ** 2)
    a_3 = 2 * beta_s + (beta_d_s + beta_d_e) * delta_t - 2 * beta_e
    a_3 /= (delta_t ** 3)
    return a_0, a_1, a_2, a_3


def calculate_psaj(coeffs, delta_t, nr_points=500):
    N = len(coeffs)
    ts = np.linspace(0, N * delta_t, nr_points)
    poses, spds, accs, jrks = [], [], [], []
    for t in ts:
        t %= (delta_t * N)  # should only happen at last point?
        t_trunc = int(t // delta_t)
        a_0, a_1, a_2, a_3 = coeffs[t_trunc]
        dt = t - (t_trunc * delta_t)
        pos, spd, acc, jrk = calculate_pose_speed_acc_jerk_from_cubic_coefficients(a_0, a_1, a_2, a_3, dt)
        poses.append(pos)
        spds.append(spd)
        accs.append(acc)
        jrks.append(jrk)
    return np.c_[poses], np.c_[spds], np.c_[accs], np.c_[jrks]


def interpolate_trajectory_from_cubic_coeffs(a_0, a_1, a_2, a_3, delta_t, nr_points):
    ts = np.linspace(0, delta_t, nr_points)
    poses, spds, accs, jrks = [], [], [], []
    for t in ts:
        pos, spd, acc, jrk = calculate_pose_speed_acc_jerk_from_cubic_coefficients(a_0, a_1, a_2, a_3, t)
        poses.append(pos)
        spds.append(spd)
        accs.append(acc)
        jrks.append(jrk)
    return np.c_[poses], np.c_[spds], np.c_[accs], np.c_[jrks]


def calculate_pose_speed_acc_jerk_from_cubic_coefficients(a_0, a_1, a_2, a_3, t):
    pos = a_0 + a_1 * t + a_2 * t ** 2 + a_3 * t ** 3
    spd = a_1 + 2 * a_2 * t + 3 * a_3 * t ** 2
    acc = 2 * a_2 + 6 * a_3 * t
    jrk = 6 * a_3
    return pos, spd, acc, jrk




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