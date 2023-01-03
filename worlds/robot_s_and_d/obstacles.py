import copy

import numpy as np
import trimesh

import pyrb
from pyrb.traj.via_point import calculate_cubic_coefficients_between_via_points, \
    interpolate_trajectory_from_cubic_coeffs
from worlds.robot_s_and_d.utils import compute_rot_matrices, compute_rotation_matrix_from_angles


class ObstacleStatic:

    def __init__(self, mesh, name=""):
        self.mesh = mesh
        self.name = name


class ObstacleDynamic:

    def __init__(self, mesh, ps, Rs, name=""):
        self.mesh = mesh
        self.positions = ps
        self.rotations = Rs
        self.name = name

    def get_transform_at_time_step(self, ts):
        i = int(ts) % self.positions.shape[0]   # TODO: need to interpolate between time steps
        p = self.positions[i, :]
        R = self.rotations[:, :, i].T
        return pyrb.kin.rot_trans_to_SE3(p=p, R=R)

    def get_swept_motion(self):
        meshes = []
        for i in range(self.positions.shape[0]):
            T = self.get_transform_at_time_step(i)
            mesh = self.mesh.copy()
            mesh.apply_transform(T)
            meshes.append(mesh)
        return trimesh.boolean.union(meshes)


class ObstacleManager:

    def __init__(self, mesh_safe_region=None):
        self.mesh_safe_region = mesh_safe_region
        self.collision_manager = trimesh.collision.CollisionManager()
        self.dynamic_obstacles = []
        self.static_obstacles = []

    #
    # def get_state_dict(self):
    #     return {
    #         "dynamic_obstacles": copy.deepcopy(self.dynamic_obstacles),
    #         "static_obstacles": copy.deepcopy(self.static_obstacles)
    #     }
    #
    # def load_state_dict(self, state):
    #     self.clear()
    #     self.dynamic_obstacles = state["dynamic_obstacles"]
    #     for o in self.dynamic_obstacles:
    #         self.collision_manager.add_object(
    #             name=o.name, mesh=o.mesh
    #         )
    #     self.static_obstacles = state["static_obstacles"]
    #     for o in self.static_obstacles:
    #         self.collision_manager.add_object(
    #             name=o.name, mesh=o.mesh
    #         )

    def __getstate__(self):
        self.clear_collision_manager()
        self.collision_manager = None
        state = copy.deepcopy(self.__dict__)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.reset_collision_manager_from_obstacles()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.reset_collision_manager_from_obstacles()

    def reset_collision_manager_from_obstacles(self):
        for o in self.dynamic_obstacles + self.static_obstacles:
            self.collision_manager.add_object(
                name=o.name, mesh=o.mesh
            )

    def clear_collision_manager(self):
        for o in self.dynamic_obstacles + self.static_obstacles:
            # TODO: should bool static objects..
            self.collision_manager.remove_object(o.name)

    def reset(self):
        self.clear()
        # TODO: make obstacle generator part of OM?
        # delta_t arbitrary?
        if self.mesh_safe_region is not None:
            self.collision_manager.add_object(mesh=self.mesh_safe_region, name="safety_region")
        self.create_dynamic_obstacles()
        self.create_static_obstacles()
        if self.mesh_safe_region is not None:
            self.collision_manager.remove_object(name="safety_region")

    def clear(self):
        self.clear_collision_manager()
        self.dynamic_obstacles.clear()
        self.static_obstacles.clear()

    def create_dynamic_obstacles(self):
        dynamic_obstacle_generator = DynamicObstacleGenerator(delta_t=0.5, obstacle_manager=self)
        cnt = 0
        max_cnt = 10
        while cnt < max_cnt:
            dynamic_obstacle = dynamic_obstacle_generator.generate_dynamic_obstacle()
            cnt += 1
            if dynamic_obstacle is not None:
                self.append_dynamic_obstacle(dynamic_obstacle)

    # collision_manager_swept = self.sweep_dynamic_obstacles()
    # if self.mesh_safe_region is not None:
    #     collision_manager_swept.add_object(mesh=self.mesh_safe_region, name="safety_region")
    # self.create_static_obstacles(collision_manager_swept)
    # def sweep_dynamic_obstacles(self):
    #     # trimesh.convex.convex_hull(obj)
    #     collision_manager_swept = trimesh.collision.CollisionManager()
    #     for i, dyn_obs in enumerate(self.dynamic_obstacles):
    #         m = dyn_obs.get_swept_motion()
    #         collision_manager_swept.add_object(name=f"sweep_volume_{i}", mesh=m)
    #     return collision_manager_swept
       # if not collision_manager_swept.in_collision_single(mesh):
        #     self.append_static_obstacle(ObstacleStatic(mesh))
    # def create_static_obstacles(self, collision_manager_swept):
    #     for cnt in range(40):
    #         p = np.random.uniform(-0.75, 0.75, size=3)
    #         a, b, g = np.random.uniform(-np.pi, np.pi, size=3)
    #         R = compute_rotation_matrix_from_angles(a, b, g)
    #         height, width, depth = np.random.uniform(.1, .5, size=3)
    #         T = pyrb.kin.rot_trans_to_SE3(R, p)
    #         mesh = trimesh.creation.box((height, width, depth))
    #         mesh.apply_transform(T)
    #         if not collision_manager_swept.in_collision_single(mesh):
    #             self.append_static_obstacle(ObstacleStatic(mesh))

    def create_static_obstacles(self):
        meshes_static = []
        nr_static_obstacles = 40
        for cnt in range(nr_static_obstacles):
            p = np.random.uniform(-0.75, 0.75, size=3)
            a, b, g = np.random.uniform(-np.pi, np.pi, size=3)
            R = compute_rotation_matrix_from_angles(a, b, g)
            height, width, depth = np.random.uniform(.1, .5, size=3)
            T = pyrb.kin.rot_trans_to_SE3(R, p)
            mesh = trimesh.creation.box((height, width, depth))
            mesh.apply_transform(T)
            meshes_static.append(mesh)
        collision_manager_static = trimesh.collision.CollisionManager()
        for i, m in enumerate(meshes_static):
            collision_manager_static.add_object(f"{i}", m)
        ts_end = self.dynamic_obstacles[0].positions.shape[0]
        remove_meshes = []
        for ts in range(ts_end):
            self.set_time(ts)
            collision, names = self.collision_manager.in_collision_other(
                collision_manager_static, return_names=True
            )
            if collision:
                for _, static_name in names:
                    if static_name not in remove_meshes:
                        collision_manager_static.remove_object(static_name)
                        remove_meshes.append(static_name)
        remove_meshes = set(map(int, remove_meshes))
        meshes_keep = set(range(nr_static_obstacles)) - remove_meshes
        for i, m in enumerate(meshes_static):
            if i in meshes_keep:
                self.append_static_obstacle(ObstacleStatic(m))

    def append_dynamic_obstacle(self, obstacle):
        name = obstacle.name or f"obstacle_dynamic_{len(self.dynamic_obstacles)}"
        self.collision_manager.add_object(
            name=name, mesh=obstacle.mesh
        )
        self.dynamic_obstacles.append(obstacle)
        if not obstacle.name:
            obstacle.name = name

    def append_static_obstacle(self, obstacle):
        name = obstacle.name or f"obstacle_static_{len(self.static_obstacles)}"
        self.collision_manager.add_object(name=name, mesh=obstacle.mesh)
        self.static_obstacles.append(obstacle)
        if not obstacle.name:
            obstacle.name = name

    def is_collision_free_time_step(self, ts, mesh):
        self.set_time(ts)
        return not self.collision_manager.in_collision_single(mesh)

    def set_time(self, ts):
        for i, o in enumerate(self.dynamic_obstacles):
            T = o.get_transform_at_time_step(ts)
            self.collision_manager.set_transform(name=o.name, transform=T)


class DynamicObstacleGenerator:

    def __init__(self, delta_t, obstacle_manager):
        self.obstacle_manager = obstacle_manager
        self.delta_t = delta_t
        self.r = 0.6
        self.dim = 3

    def generate_dynamic_obstacle(self, nr_via_points=3):
        r_c = 0.1
        h_c = 0.5
        mesh = trimesh.creation.cylinder(radius=r_c, height=h_c)
        trajectory = np.array([]).reshape((0, self.dim * 2))
        ts = 0
        via_pv_points = self.sample_via_pv_point().reshape(1, -1)
        for i in range(nr_via_points):
            if i == nr_via_points - 1:
                trajectory_segment = self.compute_segment(pv_start=via_pv_points[i], pv_end=via_pv_points[0])
                is_collision_free = self.is_segment_collision_free(mesh, trajectory_segment, ts_start=ts)
                if not is_collision_free:
                    # print("Couldn't connect end...")
                    # TODO: think of this...
                    return None
            else:
                trajectory_segment = self.generate_collision_free_segment(mesh, ts, via_pv_points[i])

            if trajectory_segment is not None:
                ts += trajectory_segment.shape[0]
                via_pv_points = np.vstack([via_pv_points, trajectory_segment[-1, :]])
                trajectory = np.vstack([trajectory, trajectory_segment[:-1, :]])
            else:
                # TODO: implement backup
                return None
        positions, speeds = trajectory[:, :self.dim], trajectory[:, self.dim:]
        Rs = compute_rot_matrices(speeds)
        return ObstacleDynamic(mesh, positions, Rs)

    @staticmethod
    def generate_position_within_ball(r, dim=2):
        point = np.ones((dim,)) * np.inf
        while np.linalg.norm(point) > r:
            point = np.random.uniform(-r, r, size=(dim,))
        return point

    @staticmethod
    def generate_position_within_annululs(r, dim=2):
        r_min = 0.2
        while True:
            point = np.random.uniform(-r, r, size=(dim,))
            dist = np.linalg.norm(point)
            if r_min < dist < r:
                return point

    def sample_position(self):
        return self.generate_position_within_annululs(self.r, dim=self.dim)

    def sample_velocity(self):
        v_min, v_max = -1, 1
        return np.random.uniform(v_min, v_max, size=(self.dim, ))

    def sample_via_pv_point(self):
        return np.hstack([self.sample_position(), self.sample_velocity()])

    def generate_collision_free_segment(self, mesh, ts_start, pv_start):
        is_collision_free = False
        cnt = 0
        max_cnt = 10
        while not is_collision_free and cnt < max_cnt:
            pv_end = self.sample_via_pv_point()
            segment = self.compute_segment(pv_start, pv_end)
            is_collision_free = self.is_segment_collision_free(mesh, segment, ts_start)
            if is_collision_free:
                return segment
            cnt += 1
        return None

    def compute_segment(self, pv_start, pv_end):
        nr_points_between_via_points = 40
        coeffs = calculate_cubic_coefficients_between_via_points(
            position_start=pv_start[:self.dim],
            velocity_start=pv_start[self.dim:],
            position_end=pv_end[:self.dim],
            velocity_end=pv_end[self.dim:],
            delta_t=self.delta_t
        )
        positions, speeds, *_ = interpolate_trajectory_from_cubic_coeffs(
            *coeffs,
            delta_t=self.delta_t,
            nr_points=nr_points_between_via_points
        )
        return np.hstack([positions, speeds])

    def is_segment_collision_free(self, mesh, segment, ts_start):
        positions, speeds = segment[:, :self.dim], segment[:, self.dim:]
        Rs = compute_rot_matrices(speeds)
        T_inv = np.eye(4)
        is_collision_free = True
        ts = ts_start
        for i in range(positions.shape[0]):
            p = positions[i, :]
            R = Rs[:, :, i].T
            T = pyrb.kin.rot_trans_to_SE3(p=p, R=R)
            mesh.apply_transform(T @ T_inv)
            T_inv = pyrb.kin.SE3_inv(T)
            is_collision_free = self.obstacle_manager.is_collision_free_time_step(ts + i, mesh)
            if not is_collision_free:
                # viz_collision(positions, T)
                break
        mesh.apply_transform(T_inv)
        return is_collision_free


if __name__ == "__main__":
    import pickle
    from worlds.robot_s_and_d.render import render_obstacle_manager_meshes
    import matplotlib.pyplot as plt
    om = ObstacleManager()
    # with open("empty_obstacles.pckl", "wb") as fp:
    #     pickle.dump(om, fp)
    # om.reset()
    # with open("obstacles.pckl", "wb") as fp:
    #     pickle.dump(om, fp)
    with open("obstacles.pckl", "rb") as fp:
        om = pickle.load(fp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ts = 0
    while True:
        render_obstacle_manager_meshes(ax, om, ts)
        plt.pause(0.1)
        ax.cla()
        ts += 1
