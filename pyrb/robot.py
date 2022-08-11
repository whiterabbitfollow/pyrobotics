import pyrb
import numpy as np

import trimesh

from typing import Optional, List


class Joint:

    def __init__(self, name, limits):
        self.name = name
        self.limits = limits
        self.angle = 0


class GeometryRectangle:

    def __init__(self, width, height, depth, direction):
        self.transform = None
        self.direction = direction  # TODO: Need better name..
        self.width = width
        self.height = height
        self.depth = depth
        self.whd = (self.width, self.height, self.depth)

    def set_transform(self, transform):
        offset = np.array([0, 0, 0], dtype=np.float32)
        offset[self.direction] = self.whd[self.direction]/2
        box_offset_transform = pyrb.kin.rot_trans_to_SE3(p=offset)
        self.transform = transform @ box_offset_transform

    def produce_mesh(self, transformed=False):
        mesh = trimesh.creation.box(extents=self.whd)
        if transformed:
            mesh.apply_transform(self.transform)
        return mesh


class Link:

    def __init__(self, name, parent_joint: Joint, child_joint: Joint, geometry: Optional[GeometryRectangle] = None):
        self.name = name
        self.parent_joint = parent_joint
        self.child_joint = child_joint
        self.geometry = geometry
        self.frame = None

    def has_geometry(self):
        return self.geometry is not None

    def get_mesh(self, apply_trf=False):
        # TODO: when will apply_trf be False??? also change name...
        if self.geometry is None:
            return None
        return self.geometry.produce_mesh(transformed=apply_trf)

    def get_mesh_transform(self):
        if self.geometry is None:
            return None
        return self.geometry.transform

    def set_frame(self, frame):
        self.frame = frame
        if self.geometry is not None:
            self.geometry.set_transform(frame)

    def set_parent_joint_angle(self, angle):
        self.parent_joint.angle = angle


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
        self.links: List[Link] = []
        self.joints: List[Joint] = []
        self.collision_manager = trimesh.collision.CollisionManager()

        for i, joint in enumerate(joints):
            self.joints.append(
                Joint(
                    name=f"joint_{i}",
                    limits=joint.get("limits", (-np.pi, np.pi))
                )
            )

        for i, link in enumerate(links):
            parent_joint = self.joints[i]
            if i + 1 == len(self.joints):
                child_joint = None  # end_effector
            else:
                child_joint = self.joints[i + 1]
            geometry = GeometryRectangle(**link["geometry"]) if link["geometry"] else None
            link = Link(
                name=f"link_{i}",
                geometry=geometry,
                parent_joint=parent_joint,
                child_joint=child_joint
            )
            if link.has_geometry():
                self.collision_manager.add_object(link.name, link.get_mesh())
            self.links.append(link)

        self.joint_limits = self.get_joint_limits()

    def get_joint_limits(self):
        return np.vstack([joint.limits for joint in self.joints])

    def set_config(self, values):
        self.config = values
        transforms = self.forward(values)
        self.update_geometries(transforms)

    def get_config(self):
        return self.config

    def update_geometries(self, transforms):
        # update frames better name...
        for link, transform, angle in zip(self.links, transforms, self.config):
            link.set_frame(transform)
            link.set_parent_joint_angle(angle)
            if link.has_geometry():
                mesh_transform = link.get_mesh_transform()
                self.collision_manager.set_transform(name=link.name, transform=mesh_transform)

    def is_configuration_feasible(self, q):
        return (self.joint_limits[:, 0] <= q).all() & (q <= self.joint_limits[:, 1]).all()
