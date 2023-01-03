import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import trimesh

import pyrb

from examples.data.manipulators import DATA_MANIPULATOR_3DOF
from pyrb.mp.base_world import BaseMPWorld
from pyrb.kin import rot_x_to_SO3, rot_y_to_SO3, rot_z_to_SO3, rot_trans_to_SE3, SE3_mul
from pyrb.mp.base_agent import MotionPlanningAgent
from worlds.common import WorldData3D


def compute_rotation_matrix_from_angles(alpha, beta, gamma):
    return rot_z_to_SO3(gamma) @ rot_y_to_SO3(beta) @ rot_x_to_SO3(alpha)


def get_uniformly_sampled_points():
    ps = np.random.uniform(-1, 1, size=3)
    pe = np.random.uniform(-1, 1, size=3)
    return ps, pe


def sample_path_with_cone_constraint():
    p_s = np.random.uniform(-.5, .5, size=3)
    R = make_cs_from_point(-p_s)
    h_start = np.linalg.norm(p_s) * 1.5
    h_end = 2 * np.linalg.norm(p_s)
    phi = np.pi / 6
    p_e_local = sample_point_within_cone(h_start, h_end, phi)
    p_e = R @ p_e_local + p_s
    d = p_e - p_s
    R = make_cs_from_point(d)
    return p_s, p_e, R


def make_cs_from_point(p):
    e_x = (p / np.linalg.norm(p)).ravel()
    e_1 = np.eye(3, 1).ravel()
    e_y = normal_vector_decomposition(e_1, e_x)
    e_y = e_y / np.linalg.norm(e_y)
    e_z = np.cross(e_x, e_y)
    R = np.hstack([e_x.reshape(-1, 1), e_y.reshape(-1, 1), e_z.reshape(-1, 1)])
    return R


def normal_vector_decomposition(u, v):
    return u - u.T @ v * v


def sample_point_within_cone(h_start, h_end, phi):
    h = np.random.uniform(h_start, h_end)
    phi_p = np.random.uniform(-phi, phi)
    r_max = h * np.tan(phi)
    r = np.random.uniform(r_max * 0.5, r_max)
    p_e_local = np.array([h, r * np.cos(phi_p), r * np.sin(phi_p)])
    return p_e_local


class OOB:

    def __init__(self, transform, width, height, depth):
        self.transform = transform
        self.width = width
        self.height = height
        self.depth = depth

    def to_mesh(self):
        mesh = trimesh.creation.box(extents=(self.width, self.height, self.depth))
        mesh.apply_transform(self.transform)
        return mesh


class StaticCylinderObstacle:
    def __init__(self, dim, R, p):
        self.T = pyrb.kin.rot_trans_to_SE3(R, p)
        self.dim = dim
        self.foot_print_oob = None

    def set_time(self, t):
        pass

    def get_mesh(self):
        mesh = trimesh.creation.box(extents=self.dim)
        mesh.apply_transform(self.T)
        return mesh


class MovingCylinderObstacle:

    def __init__(self):
        self.R = None
        self.t = 0
        self.radius = 0.1
        self.height = 0.4
        self.vel = 0.1
        self.foot_print_oob = None

    def set_orientation(self, orientation):
        self.R = orientation

    def set_start_and_end(self, pos_s, pos_e):
        self.pos_s = pos_s
        self.pos_e = pos_e

    def set_foot_print_oob(self, oob):
        self.foot_print_oob = oob

    def get_mesh(self):
        mesh = trimesh.creation.cylinder(self.radius, height=self.height)
        pos = self.calculate_pos()
        mesh.apply_transform(rot_trans_to_SE3(p=pos, R=self.R))
        return mesh

    def calculate_pos(self):
        distance_points = np.linalg.norm(self.pos_e - self.pos_s)
        max_distance = 2 * distance_points
        T = max_distance / self.vel
        t = self.t % T
        distance = self.vel * t
        if t < T/2:
            s = self.pos_s
            e = self.pos_e
        else:
            s = self.pos_e
            e = self.pos_s
            distance = distance % distance_points
        beta = distance / distance_points
        pos = (1-beta) * s + beta * e
        return pos

    def set_time(self, t):
        self.t = t

    def reset(self):
        self.t = 0


class Obstacles:
    def __init__(self):
        self._obstacles = []
        self.collision_manager = trimesh.collision.CollisionManager()

    def clear(self):
        self._obstacles.clear()
        for i in range(len(self._obstacles)):
            self.collision_manager.remove_object(f"mesh_{i}")

    def append(self, obst):
        self._obstacles.append(obst)
        self.collision_manager.add_object(f"mesh_{len(self._obstacles)}", obst.get_mesh())

    def __iter__(self):
        return iter(self._obstacles)

    def set_time(self, t):
        for obst in self._obstacles:
            obst.set_time(t)


class RobotCylinderWorld(BaseMPWorld):

    def __init__(self):
        self.max_nr_obstacle = 10
        data = WorldData3D((-1, 1), (-1, 1), (-1, 1))
        robot = MotionPlanningAgent(copy.deepcopy(DATA_MANIPULATOR_3DOF))
        super().__init__(robot=robot, data=data, obstacles=Obstacles())
        transform = pyrb.kin.rot_trans_to_SE3(p=np.array([0.0, 0.0, 0.25]))
        self.patches = []
        self.safe_region_radius = 0.1
        self.safe_region_mesh = trimesh.creation.cylinder(self.safe_region_radius, height=.5, transform=transform)
        self.start_config = None
        self.goal_config = None
        self.fig_ax = None

    def reset(self):
        self.obstacles.clear()
        # np.random.randint(1, self.max_nr_obstacle)
        self.start_config = self.sample_feasible_config()
        self.robot.set_config(self.start_config)
        self.goal_config = self.sample_feasible_config()
        cnt = 0
        cd_manager = trimesh.collision.CollisionManager()
        cd_manager.add_object("safe_region", self.safe_region_mesh)
        for _ in range(self.max_nr_obstacle):
            obst = MovingCylinderObstacle()
            # p_s, p_e, R = create_motion_from_sampled_points()
            p_s, p_e, R = sample_path_with_cone_constraint()
            # p_s = np.random.uniform(self.data.lower, self.data.upper, size=(3,))
            # euler_angles = np.random.uniform(-np.pi, np.pi, size=(3,))
            # distance = np.random.uniform(0.5, np.sqrt(2) * 2)
            r = obst.radius
            # l = distance
            l = np.linalg.norm(p_e - p_s)
            width = 2 * r + l
            height = 2 * r
            depth = obst.height
            #
            # R = compute_rotation_matrix_from_angles(*euler_angles)
            # T = rot_trans_to_SE3(R, p_s)
            # p_e_local = np.eye(3, 1) * distance
            # p_e = SE3_mul(T, p_e_local)
            # p_e = p_e.ravel()
            p_m = (p_e + p_s) / 2
            T_footprint = rot_trans_to_SE3(R, p_m)
            foot_print_OOB = OOB(T_footprint, width, height, depth)
            fp_mesh = foot_print_OOB.to_mesh()
            if not cd_manager.in_collision_single(fp_mesh):
                # cd_manager.add_object(f"obstacle_foot_print_{cnt}", fp_mesh)
                obst.set_orientation(R)
                obst.set_start_and_end(p_s, p_e)
                obst.set_foot_print_oob(foot_print_OOB)
                self.obstacles.append(obst)
                cnt += 1

        # for cnt in range(20):
        #     p = np.random.uniform(-0.75, 0.75, size=3)
        #     a, b, g = np.random.uniform(-np.pi, np.pi, size=3)
        #     R = compute_rotation_matrix_from_angles(a, b, g)
        #     height, width, depth = np.random.uniform(.1, .5, size=3)
        #     T = pyrb.kin.rot_trans_to_SE3(R, p)
        #     mesh = trimesh.creation.box((height, width, depth))
        #     mesh.apply_transform(T)
        #     if not cd_manager.in_collision_single(mesh):
        #         obst = StaticCylinderObstacle((height, width, depth), R, p)
        #         cd_manager.add_object(f"static_object_{cnt}", mesh)
        #         self.obstacles.append(obst)
        #         cnt += 1

    def render_world(self, ax):
        curr_state = self.robot.config
        goal_config = self.goal_config
        if goal_config is not None:
            self.robot.set_config(goal_config)
            for link in self.robot.links:
                mesh = link.get_mesh(apply_trf=True)
                if mesh is None:
                    continue
                ax.plot_trisurf(
                    mesh.vertices[:, 0],
                    mesh.vertices[:, 1],
                    triangles=mesh.faces,
                    Z=mesh.vertices[:, 2],
                    color="blue",
                    alpha=0.1
                )
            self.robot.set_config(curr_state)
        is_collision = False
        agent_color = "red" if is_collision else "blue"
        for link in self.robot.links:
            mesh = link.get_mesh(apply_trf=True)
            if mesh is None:
                continue
            alpha = 1.0
            # if observer is not None:
            #     alpha = 0.1
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                triangles=mesh.faces,
                Z=mesh.vertices[:, 2],
                color=agent_color,
                alpha=alpha
            )

        mesh = self.safe_region_mesh
        alpha = 0.1
        # if observer is not None:
        #     alpha = 0.1
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            triangles=mesh.faces,
            Z=mesh.vertices[:, 2],
            color=agent_color,
            alpha=alpha
        )

        for obst in self.obstacles:
            alpha = 1.0
            mesh = obst.get_mesh()  # apply_trf=False
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                triangles=mesh.faces,
                Z=mesh.vertices[:, 2],
                color=agent_color,
                alpha=alpha
            )
            if obst.foot_print_oob is not None:
                mesh = obst.foot_print_oob.to_mesh()  # apply_trf=False
                ax.plot_trisurf(
                    mesh.vertices[:, 0],
                    mesh.vertices[:, 1],
                    triangles=mesh.faces,
                    Z=mesh.vertices[:, 2],
                    color="red",
                    alpha=0.1
                )
                line = np.vstack([obst.pos_s.reshape(1, -1), obst.pos_e.reshape(1, -1)])
                ax.plot(line[:, 0], line[:, 1], line[:, 2])

        ax.set_xlim(*self.data.x.to_tuple())
        ax.set_ylim(*self.data.y.to_tuple())
        ax.set_zlim(*self.data.z.to_tuple())
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_zaxis().set_ticks([])

    def view(self):
        if self.fig_ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.fig_ax = (fig, ax)
        else:
            (fig, ax) = self.fig_ax
        self.render_world(ax)
        plt.show()

    def is_collision_free_state(self, state) -> bool:
        self.robot.set_config(state)
        return False    # np.linalg.norm(self.obstacles.pos - state) > self.obstacles.radius

    def set_time(self, t):
        self.t = t
        self.obstacles.set_time(t)


if __name__ == "__main__":
    world = RobotCylinderWorld()
    world.reset()
    # world.robot.set_config(np.array([0.3, 3]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t in range(1000):
        world.set_time(t)
        world.render_world(ax)
        plt.pause(0.1)
        ax.cla()
