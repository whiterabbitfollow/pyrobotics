import pyrb
import numpy as np
import trimesh
import time
import copy

from examples.data.manipulators import DATA_MANIPULATOR_2DOF, DATA_MANIPULATOR_3DOF
from pyrb.traj import via_point


class MovingRobotAdversary(pyrb.robot.Manipulator):

    def __init__(
            self,
            robot_data,
            trajectory=None,
            reset_mode="random"
    ):
        super().__init__(robot_data)
        self.is_static = False
        self.time_step = 0
        self.decimal_part = 0
        assert reset_mode in ("random", "fixed")
        self.reset_mode = reset_mode
        if self.reset_mode == "fixed":
            assert trajectory is not None, "No trajectory set"
            self.traj = trajectory
        else:
            self.traj = np.array([])
        self.joint_limits = self.get_joint_limits()
        # cylinder_radii = 0.1
        # self.invalid_region_mesh = trimesh.creation.cylinder(cylinder_radii, height=0.1)
        self.invalid_region_mesh = trimesh.creation.box(extents=(0.2, 1.0, 0.2))
    def set_time(self, time_step):
        self.time_step = int(self.truncate_time_step(time_step))
        self.decimal_part = time_step - int(time_step)
        if self.decimal_part == 0:
            config = self.traj[self.time_step]
            self.set_config(config)
        else:
            self.interpolate_trajectory_between_time_steps(self.decimal_part)

    def get_complete_time(self):
        return self.time_step + self.decimal_part

    def interpolate_trajectory_between_time_steps(self, decimal_part):
        t_curr = self.time_step
        t_nxt = self.truncate_time_step(self.time_step + 1)
        self.set_config(self.interpolate_linearly(self.traj[t_curr, :], self.traj[t_nxt, :], decimal_part))

    def interpolate_linearly(self, x1, x2, frac):
        assert 0.0 <= frac <= 1.0
        return (1 - frac) * x1 + frac * x2

    def truncate_time_step(self, time_step):
        return time_step % self.traj.shape[0]

    def reset(self, seed=None, traj=None):
        if traj is not None:
            self.traj = traj
        elif self.reset_mode == "random":
            max_nr_retries = 10
            nr_retries = 0
            while True:
                try:
                    self.set_random_trajectory(seed)
                    break
                except:
                    nr_retries += 1
                if nr_retries > max_nr_retries:
                    if seed is not None:
                        raise RuntimeError()
                    else:
                        print(f"Warning!!! nr retires for finding a traj {nr_retries}, seed {seed}")
        else:
            assert self.traj.shape[0] > 0, "No trajectory set"

    def set_random_trajectory(self, seed=None):
        state = None
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed)
        nr_joints = self.joint_limits.shape[0]
        pose_first = self.sample_valid_pose()
        velocity_first = np.random.uniform(0, 2 * np.pi, nr_joints)

        max_nr_via_points = 10
        error_nr_via_points = 12
        nr_via_points = 1
        poses = []

        pose_start = pose_first
        velocity_start = velocity_first

        time_s = time.time()
        time_limit = 3

        while True:
            pose_next = self.sample_valid_pose()
            velocity_next = np.random.uniform(0, 2 * np.pi, nr_joints)
            segment_poses = self.create_segment_between_via_points(
                pose_start,
                velocity_start,
                pose_next,
                velocity_next
            )
            if segment_poses is not None:
                if nr_via_points != 1:
                    segment_poses = segment_poses[1:, :]
                poses.append(segment_poses)
                nr_via_points += 1
                pose_start = pose_next
                velocity_start = velocity_next

            if nr_via_points >= max_nr_via_points:
                segment_poses_last = self.create_segment_between_via_points(
                    pose_start, velocity_start,
                    pose_first, velocity_first
                )

                if segment_poses_last is not None:
                    poses.append(segment_poses_last[1:-1])
                    break
            time_e = time.time() - time_s
            if nr_via_points >= error_nr_via_points or time_e > time_limit:
                raise RuntimeError("Couldn't create a valid trajectory")

        self.traj = np.vstack(poses)
        if state is not None:
            np.random.set_state(state)

    def sample_valid_pose(self):
        is_collision = True
        pose = None
        while is_collision:
            pose = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
            is_collision = self.is_pose_in_collision(pose)
        return pose

    def create_segment_between_via_points(self, pose_start, velocity_start, pose_next, velocity_next):
        delta_t = 0.1
        a_0, a_1, a_2, a_3 = via_point.calculate_cubic_coefficients_between_via_points(
            pose_start, velocity_start, pose_next, velocity_next, delta_t
        )
        segment_poses, spds, accs, jrks = via_point.interpolate_trajectory_from_cubic_coeffs(
            a_0, a_1, a_2, a_3,
            delta_t,
            nr_points=70
        )
        if self.validate_poses(segment_poses):
            return segment_poses
        else:
            return None

    def validate_poses(self, poses):
        is_valid = True
        for config in poses:
            is_collision = self.is_pose_in_collision(config)
            if is_collision:
                is_valid = False
                break
        return is_valid

    def is_pose_in_collision(self, config):
        self.set_config(config)
        return self.collision_manager.in_collision_single(self.invalid_region_mesh)


class MobileMovingRobotAdversary(MovingRobotAdversary):

    def __init__(self, base_nr_dof, *args, **kwargs):
        self.base_transformation = pyrb.kin.rot_trans_to_SE3(p=np.array([0, 0, 0]))
        self.base_nr_dof = base_nr_dof
        super().__init__(*args, **kwargs)

    def set_config(self, values):
        self.base_position = values[:self.base_nr_dof]
        super().set_config(values[self.base_nr_dof:])

    def get_config(self):
        return np.hstack([self.base_position, self.config])

    def forward(self, angles, include_ee=False):
        self.base_transformation = pyrb.kin.rot_trans_to_SE3(p=self.base_position)
        transforms_local = super().forward(angles)
        transforms = [self.base_transformation @ trf for trf in transforms_local]
        return transforms


class Mobile2DOFAdversaryManipulator(MobileMovingRobotAdversary):

    def __init__(self):
        joint_limits = np.array([
            [0.1, .4],
            [-.4, .4],
            [0, 2 * np.pi],
            [0, 2 * np.pi]
        ])
        robot_data = copy.deepcopy(DATA_MANIPULATOR_2DOF)
        robot_data["joints"][0]["limits"] = joint_limits[2]
        robot_data["joints"][1]["limits"] = joint_limits[3]
        super().__init__(base_nr_dof=2, robot_data=robot_data)
        self.joint_limits = joint_limits    # TODO: very buggy... needs to be set in order for sampling of config to work...


class Mobile3DOFAdversaryManipulator(MobileMovingRobotAdversary):

    def __init__(self):
        joint_limits = np.array([
            [-.3, .3],
            [-.3, .3],
            [-.5, .5],
            [0, 2 * np.pi],
            [0, 2 * np.pi],
            [0, 2 * np.pi]
        ])
        robot_data = copy.deepcopy(DATA_MANIPULATOR_3DOF)
        robot_data["joints"][0]["limits"] = joint_limits[3]
        robot_data["joints"][1]["limits"] = joint_limits[4]
        robot_data["joints"][2]["limits"] = joint_limits[5]
        super().__init__(base_nr_dof=3, robot_data=robot_data)
        self.joint_limits = joint_limits    # TODO: very buggy... needs to be set in order for sampling of config to work...
