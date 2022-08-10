import numpy as np

from pyrb.mp.base_agent import MotionPlanningAgentActuated


class Manipulator2DOF(MotionPlanningAgentActuated):

    def __init__(self, joint_limits=None):
        max_actuation = 0.1
        if joint_limits is None:
            joint_limits = [
                [-2 * np.pi / 3, 2 * np.pi / 3],
                [-np.pi + np.pi / 4, 0]
            ]
        robot_data = {
            "links": [
                {
                    "geometry": {
                        "width": 0.3,
                        "height": 0.1,
                        "depth": 0.1,
                        "direction": 0
                    }
                },
                {
                    "geometry": {
                        "width": 0.3,
                        "height": 0.1,
                        "depth": 0.1,
                        "direction": 0
                    }
                }
            ],
            "joints": [
                {
                    "position": np.array([0.0, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": joint_limits[0]
                },
                {
                    "position": np.array([0.3, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": joint_limits[1]
                }
            ],
            "end_effector": {
                "position": np.array([0.3, 0.0, 0.0])
            }
        }
        super().__init__(robot_data=robot_data, max_actuation=max_actuation)


class Manipulator1DOF(MotionPlanningAgentActuated):

    def __init__(self):
        max_actuation = 0.1
        robot_data = {
            "links": [
                {
                    "geometry": {
                        "width": 0.3,
                        "height": 0.1,
                        "depth": 0.1,
                        "direction": 0
                    }
                }
            ],
            "joints": [
                {
                    "position": np.array([0.0, 0.0, 0.0]),
                    "rotation": np.array([0.0, 0.0, 1.0]),
                    "limits": [-np.pi/2, np.pi/2]
                },
            ],
            "end_effector": {
                "position": np.array([0.3, 0.0, 0.0])
            }
        }
        super().__init__(robot_data=robot_data, max_actuation=max_actuation)

