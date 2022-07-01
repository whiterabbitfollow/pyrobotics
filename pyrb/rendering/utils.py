import pyrb
import numpy as np


def robot_configuration_to_matplotlib_rectangles(robot):
    boxes = []
    for i, link in enumerate(robot.links):
        if not link.has_geometry():
            continue
        geometry = link.geometry
        width = geometry.width
        height = geometry.height
        box_corner_lower_local = np.array([0, -height/2, 0])
        x, y, _ = pyrb.kin.SE3_mul(link.frame, box_corner_lower_local)
        angle = np.rad2deg(pyrb.kin.angle_from_SE3_rot_z(link.frame))
        boxes.append(((x, y), width, height, angle))
    return boxes

