import matplotlib.pyplot as plt
import numpy as np

import pyrb

joint_limits = [
    [-np.pi / 2, np.pi / 2],
    [-np.pi / 2, np.pi / 2],
    [-np.pi + np.pi / 4, 0]  # TODO: set -np.pi
]

robot_data = {
    "links": [
        {
            "geometry": None
        },
        {
            "geometry": {
                "width": 0.1,
                "height": 0.1,
                "depth": 0.3
            }
        },
        {
            "geometry": {
                "width": 0.1,
                "height": 0.1,
                "depth": 0.3
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
            "position": np.array([0.0, 0.0, 0.0]),
            "rotation": np.array([0.0, 1.0, 0.0]),
            "limits": joint_limits[1]
        },
        {
            "position": np.array([0.0, 0.0, 0.3]),
            "rotation": np.array([0.0, 1.0, 0.0]),
            "limits": joint_limits[2]
        }
    ],
    "end_effector": {
        "position": np.array([0.0, 0.0, 0.3])
    }
}

robot = pyrb.robot.Manipulator(robot_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    robot.set_config(np.array([np.random.random(), np.random.random(), np.random.random()]))
    curr_state = robot.angles
    for link in robot.links:
        mesh = link.get_mesh(apply_trf=True)
        if mesh is None:
            continue
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            triangles=mesh.faces,
            Z=mesh.vertices[:, 2],
            color="blue"
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.pause(.1)
    ax.cla()
