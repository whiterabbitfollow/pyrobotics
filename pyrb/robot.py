import pyrb
import numpy as np

import trimesh


class Link:

    def __init__(self, name, geometry):
        self.name = name
        self.width = geometry["width"]
        self.height = geometry["height"]
        # TODO: only 2D so far...
        self.transform = None

    def get_geometry(self, apply_trf=False):
        geometry = trimesh.creation.box(extents=(self.width, self.height, self.height))
        if apply_trf:
            box_offset = pyrb.kin.rot_trans_to_SE3(p=np.array([self.width/2, 0, 0]))
            transform = self.transform @ box_offset
            geometry.apply_transform(transform)
        return geometry


class Joint:

    def __init__(self, name, limits):
        self.name = name
        self.limits = limits


class Manipulator(pyrb.kin.SerialKinematicChain):
    """
    robot_data = {
        "links": [
            {
                "geometry": 0.1
            },
            {
                "geometry": 0.1
            }
        ],
        "joints": [
            {
                "position": np.array([0.0, 0.0, 0.0]),
                "rotation": np.array([0.0, 0.0, 1.0])
            },
            {
                "position": np.array([1.0, 0.0, 0.0]),
                "rotation": np.array([0.0, 0.0, 1.0])
            }
        ],
        "end_effector": {
            "position": np.array([1.0, 0.0, 0.0])
        }
    }
    """

    def __init__(self, robot_data):
        links, joints = robot_data["links"], robot_data["joints"]
        kinematics = [(j["position"], j["rotation"]) for j in joints]
        end_effector_position = robot_data.get("end_effector", {}).get("position", np.zeros((3,)))
        kinematics.append((end_effector_position, None))
        super().__init__(kinematics)
        self.links, self.joints = [], []
        self.collision_manager = trimesh.collision.CollisionManager()

        for i, link in enumerate(links):
            link = Link(name=f"link_{i}", geometry=link["geometry"])
            self.collision_manager.add_object(link.name, link.get_geometry())
            self.links.append(link)

        for i, joint in enumerate(joints):
            self.joints.append(
                Joint(
                    name=f"joint_{i}",
                    limits=joint.get("limits", (-np.pi, np.pi))
                )
            )

    def get_joint_limits(self):
        return np.vstack([joint.limits for joint in self.joints])

    def set_config(self, values):
        self.config = values
        transforms = self.forward(values)
        self.update_geometries(transforms)

    def update_geometries(self, transforms):
        for link, transform in zip(self.links, transforms):
            link.transform = transform
            box_offset = pyrb.kin.rot_trans_to_SE3(p=np.array([link.width / 2, 0, 0]))
            collision_trf = transform @ box_offset
            self.collision_manager.set_transform(name=link.name, transform=collision_trf)

