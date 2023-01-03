import copy
import trimesh
import pickle

from examples.data.manipulators import DATA_MANIPULATOR_3DOF
from pyrb.mp.base_agent import MotionPlanningAgent, MotionPlanningAgentActuated
from pyrb.mp.base_world import BaseMPWorld, BaseMPTimeVaryingWorld
from worlds.common import WorldData3D
from worlds.robot_s_and_d.obstacles import ObstacleManager


class RobotCylinderWorld(BaseMPTimeVaryingWorld):

    def __init__(self):
        data = WorldData3D((-1, 1), (-1, 1), (-1, 1))
        robot = MotionPlanningAgentActuated(copy.deepcopy(DATA_MANIPULATOR_3DOF), max_actuation=0.1)
        # transform = pyrb.kin.rot_trans_to_SE3(p=np.array([0.0, 0.0, 0.25]))
        # self.mesh = trimesh.creation.cylinder(safe_region_radius, height=.5, transform=transform)
        safe_region_radius = 0.1
        mesh_safe_region = trimesh.creation.icosphere(radius=safe_region_radius)
        super().__init__(robot=robot, data=data, obstacles=ObstacleManager(mesh_safe_region=mesh_safe_region))

    def reset(self):
        self.obstacles.reset()
        self.set_time(0)

    def view(self):
        pass

    def save_state(self, file_name):
        with open(file_name, "wb") as fp:
            pickle.dump(self.obstacles, fp)

    def load_state(self, file_name):
        with open(file_name, "rb") as fp:
            self.obstacles = pickle.load(fp)
        self.set_time(0)
