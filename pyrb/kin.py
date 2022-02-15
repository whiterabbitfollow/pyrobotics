import numpy as np


def forward(kinematic_chain):
    T = np.eye(4)
    all_cs = []
    for screw_vec, theta, cs in kinematic_chain:
        T = T @ SE3_exp(screw_vec, theta)
        all_cs.append(T @ cs)
    return all_cs


def angle_from_SE3_rot_z(T):
    return np.arctan2(T[1, 0], T[0, 0])


def SE3_mul(T, p):
    return T[0:3, 0:3] @ p + T[0:3, 3]


def rot_trans_to_SE3(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[0:3, 0:3] = R
    if p is not None:
        p = p.ravel()
        T[0: p.shape[0], 3] = p
    return T


def rot_z_to_SO3(angle):
    R = np.eye(3)
    R[0, 0] = np.cos(angle)
    R[1, 1] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    return R


def vec_to_skew(w_vec):
    w_vec = w_vec.reshape(-1)
    w_skew = np.zeros((3, 3))
    w_skew[0, 1] = -w_vec[2]
    w_skew[1, 0] = w_vec[2]
    w_skew[0, 2] = w_vec[1]
    w_skew[2, 0] = -w_vec[1]
    w_skew[1, 2] = -w_vec[0]
    w_skew[2, 1] = w_vec[0]
    return w_skew


def g_theta(w_skew, theta):
    return np.eye(3) * theta + (1 - np.cos(theta)) * w_skew + (theta - np.sin(theta)) * w_skew.dot(w_skew)


def SO3_exp(w_skew, theta):
    return np.eye(3) + w_skew * np.sin(theta) + (1 - np.cos(theta)) * w_skew.dot(w_skew)


def SE3_exp(screw_vec, theta):
    w, v = screw_vec[0:3], screw_vec[3::]
    w_skew = vec_to_skew(w)
    g = g_theta(w_skew, theta)
    p = g.dot(v)
    R = SO3_exp(w_skew, theta)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def rot_trans_to_SE3(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[0:3, 0:3] = R
    if p is not None:
        p = p.ravel()
        T[0: p.shape[0], 3] = p
    return T


class KinematicChain:

    def __init__(self, kinematics):
        self.nr_joints = len(kinematics) - 1
        self.config = np.zeros((self.nr_joints,))
        self.frames = np.c_[[np.eye(4) for _ in range(self.nr_joints + 1)]]
        self.screws = np.zeros((self.nr_joints, 6))
        self.initialize_parts(kinematics)

    def initialize_parts(self, kinematics):
        kin_vec_total = np.zeros((3,))
        for i, (pos_rel_parent, rot_vec) in enumerate(kinematics[:-1]):
            kin_vec_total += pos_rel_parent
            self.frames[i, 0:3, 3] = kin_vec_total
            self.screws[i, 0:3] = rot_vec
            self.screws[i, 3:] = np.cross(kin_vec_total, rot_vec)
        pos_ee, *_ = kinematics[-1]
        self.frames[-1, :3, 3] = kin_vec_total + pos_ee

    def set_config(self, values):
        self.config = values
        self.transforms = self.forward(values)

    def forward(self, angles, include_ee=False):
        nr_frames = self.nr_joints + 1 if include_ee else self.nr_joints
        kinematic_chain = [
            (self.screws[i, :], angles[i], self.frames[i, :, :])
            if i != self.nr_joints else
            (None, None, self.frames[i, :, :])
            for i in range(nr_frames)
        ]
        T = np.eye(4)
        all_cs = []
        for screw_vec, theta, cs in kinematic_chain:
            if screw_vec is not None:
                T = T @ SE3_exp(screw_vec, theta)
            all_cs.append(T @ cs)
        return all_cs


if __name__ == "__main__":
    kinematics = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([1.0, 0.0, 0.0]), None)
    ]
    chain = KinematicChain(kinematics)
    ts = chain.forward(np.array([-np.pi/2, 0]), include_ee=True)

    for t in ts:
        print("")
        print(t)

    # kinematics = [
    #     (np.array([0.0, 0.1, 0.0]), np.array([0.0, 1.0, 0.0])),
    #     (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    #     (np.array([0.3, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    #     (np.array([0.3, 0.0, 0.0]), None)
    # ]
    # link_geometries = [
    #     (np.array([0.1, 0.1, 0.1]), np.array([0.0, 0.1, 0.0])),
    #     (np.array([0.3, 0.1, 0.1]), np.array([0.3, 0.0, 0.0])),
    #     (np.array([0.3, 0.1, 0.1]), np.array([0.3, 0.0, 0.0])),
    # ]
    # chain = KinematicChain(kinematics)
    # ts = chain.forward(np.array([np.pi/2, 0.0, 0.0]))
    # for t in ts:
    #     print(t)