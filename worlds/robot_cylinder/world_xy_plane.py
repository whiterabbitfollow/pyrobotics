import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import trimesh

import pyrb

from examples.data.manipulators import DATA_MANIPULATOR_3DOF
from pyrb.mp.base_world import BaseMPWorld
from worlds.common import WorldData3D
from pyrb.mp.base_agent import MotionPlanningAgent


def cylinder_straight_line_to_2D_OOB(p_s, p_e, r):
    # Oriented Bounding Boxes
    d = (p_e - p_s).reshape(-1, 1)
    l = np.linalg.norm(d)
    width = 2 * r + l
    height = 2 * r
    dn = d / l
    theta = np.arctan2(dn[1, 0], dn[0, 0])
    p_cs = ((p_s + p_e) / 2).reshape(-1, 1)
    return p_cs, theta, width, height


def OOB_2D_to_matplotlib_rectangle(p_cs, theta, width, height):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    theta_deg = np.rad2deg(theta)
    R = np.array([[c_theta, -s_theta], [s_theta, c_theta]])
    p_local_lower_left_box_cs = np.array([-width/2, -height/2]).reshape((2, 1))
    p_global = R @ p_local_lower_left_box_cs + p_cs[:2].reshape((2, 1))
    return Rectangle(p_global.ravel()[:2], width=width, height=height, angle=theta_deg, color="red", alpha=0.1)


class OOB2D:

    def __init__(self, p_2d, theta, width, height):
        self.p_2d = p_2d
        self.theta = theta
        self.width = width
        self.height = height

    def to_mesh(self):
        p_cs = np.append(self.p_2d.ravel(), 0)
        R = pyrb.kin.rot_z_to_SO3(angle=self.theta)
        transform = pyrb.kin.rot_trans_to_SE3(R, p_cs)
        mesh = trimesh.creation.box(extents=(self.width, self.height, 0.1))
        mesh.apply_transform(transform)
        return mesh


class CylinderObstacle:

    def __init__(self):
        self.t = 0
        self.radius = 0.1
        self.height = 0.4
        self.vel = 0.1
        self.foot_print_oob = None

    def set_start_and_end(self, pos_s, pos_e):
        self.pos_s = pos_s
        self.pos_e = pos_e

    def set_foot_print_oob(self, oob):
        self.foot_print_oob = oob

    def get_mesh(self):
        transform = pyrb.kin.rot_trans_to_SE3(p=np.array([0.0, 0.0, self.height/2]))
        mesh = trimesh.creation.cylinder(self.radius, height=self.height, transform=transform)
        pos = self.calculate_pos()
        mesh.apply_transform(pyrb.kin.rot_trans_to_SE3(p=pos))
        return mesh

    def calculate_pos(self):
        distance = self.vel * self.t
        max_distance = np.linalg.norm(self.pos_e - self.pos_s)
        distance = distance % max_distance
        beta = distance / max_distance
        pos = (1-beta) * self.pos_s + beta * self.pos_e
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

    def append(self, obst: CylinderObstacle):
        self._obstacles.append(obst)
        self.collision_manager.add_object(f"mesh_{len(self._obstacles)}", obst.get_mesh())

    def __iter__(self):
        return iter(self._obstacles)


class RobotCylinderWorld(BaseMPWorld):

    def __init__(self):
        self.max_nr_obstacle = 10
        data = WorldData3D((-1, 1), (-1, 1), (-1, 1))
        robot = MotionPlanningAgent(copy.deepcopy(DATA_MANIPULATOR_3DOF))
        super().__init__(robot=robot, data=data, obstacles=Obstacles())
        transform = pyrb.kin.rot_trans_to_SE3(p=np.array([0.0, 0.0, 0.5]))
        self.patches = []
        self.safe_region_radius = 0.1
        self.safe_region_mesh = trimesh.creation.cylinder(self.safe_region_radius, height=1.0, transform=transform)
        self.start_config = None
        self.goal_config = None

    def reset(self):
        cd_manager = trimesh.collision.CollisionManager()
        self.obstacles.clear()
        origin = np.zeros((3, ))
        obst_poses_start = np.array(origin).reshape((1, 3))
        obst_poses_end = np.array(origin).reshape((1, 3))
        obst_radiis = np.array([self.safe_region_radius])
        # np.random.randint(1, self.max_nr_obstacle)
        cd_manager.add_object("safe_region", self.safe_region_mesh)
        cnt = 0
        for _ in range(self.max_nr_obstacle):
            obst = CylinderObstacle()
            pos_start = self.generate_free_obst_pos(obst_poses_start, obst_radiis)
            pos_end = self.generate_free_obst_pos(obst_poses_start, obst_radiis)
            p_cs_2d, theta, width, height = cylinder_straight_line_to_2D_OOB(pos_start[:2], pos_end[:2], obst.radius)
            foot_print_OOB2D = OOB2D(p_cs_2d, theta, width, height)
            fp_mesh = foot_print_OOB2D.to_mesh()
            if not cd_manager.in_collision_single(fp_mesh):
                cd_manager.add_object(f"obstacle_foot_print_{cnt}", fp_mesh)
                obst_poses_start = np.vstack([obst_poses_end, obst_poses_start])
                obst_radiis = np.append(obst_radiis, obst.radius)
                obst.set_start_and_end(pos_start, pos_end)
                obst.set_foot_print_oob(foot_print_OOB2D)
                self.obstacles.append(obst)
                cnt += 1
        self.start_config = self.sample_feasible_config()
        self.robot.set_config(self.start_config)
        self.goal_config = self.sample_feasible_config()

    def generate_free_obst_pos(self, occupied_obst_poses, occupied_obst_radiis):
        while True:
            pos = np.random.uniform(self.data.lower[:2], self.data.upper[:2])
            pos = np.append(pos, 0)
            if (np.linalg.norm(occupied_obst_poses - pos, axis=1) > occupied_obst_radiis * 2).all():
                break
        return pos

    def view(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # self.fig_ax = (fig, ax)
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
        plt.show()

    def is_collision_free_state(self, state) -> bool:
        self.robot.set_config(state)
        return False    # np.linalg.norm(self.obstacles.pos - state) > self.obstacles.radius


if __name__ == "__main__":
    world = RobotCylinderWorld()
    world.reset()
    # world.robot.set_config(np.array([0.3, 3]))
    world.view()
