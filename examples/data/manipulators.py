import numpy as np


DATA_MANIPULATOR_3DOF = {
    "links": [
        {
            "geometry": None
        },
        {
            "geometry": {
                "width": 0.1,
                "height": 0.1,
                "depth": 0.5,
                "direction": 2
            }
        },
        {
            "geometry": {
                "width": 0.1,
                "height": 0.1,
                "depth": 0.5,
                "direction": 2
            }
        }
    ],
    "joints": [
        {
            "position": np.array([0.0, 0.0, 0.0]),
            "rotation": np.array([0.0, 0.0, 1.0]),
            "limits": [-np.pi / 2, np.pi / 2]
        },
        {
            "position": np.array([0.0, 0.0, 0.0]),
            "rotation": np.array([0.0, 1.0, 0.0]),
            "limits": [-np.pi / 2, np.pi / 2]
        },
        {
            "position": np.array([0.0, 0.0, 0.5]),
            "rotation": np.array([0.0, 1.0, 0.0]),
            "limits": [-np.pi + np.pi / 4, 0]
        }
    ],
    "end_effector": {
        "position": np.array([0.0, 0.0, 0.5])
    }
}

DATA_MANIPULATOR_2DOF = {
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
            "limits": [-2 * np.pi / 3, 2 * np.pi / 3]
        },
        {
            "position": np.array([0.3, 0.0, 0.0]),
            "rotation": np.array([0.0, 0.0, 1.0]),
            "limits": [-np.pi + np.pi / 4, 0]
        }
    ],
    "end_effector": {
        "position": np.array([0.3, 0.0, 0.0])
    }
}

DATA_MANIPULATOR_1DOF = {
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
            "limits": [-(3 * np.pi) / 2, (3 * np.pi) / 2]
        },
    ],
    "end_effector": {
        "position": np.array([0.3, 0.0, 0.0])
    }
}
