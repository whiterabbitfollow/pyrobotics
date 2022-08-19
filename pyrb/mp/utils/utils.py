import time

import numpy as np




def start_timer():
    time_s = time.time()
    time_elapsed = time.time() - time_s
    return time_s, time_elapsed


def is_vertex_in_goal_region(state, state_goal, goal_region_radius):
    distance = np.linalg.norm(state - state_goal)
    return distance < goal_region_radius


class Status:
    SUCCESS = "success"
    FAILURE = "failure"


def compile_planning_data(path, time_elapsed, nr_verts):
    status = Status.SUCCESS if path.size else Status.FAILURE
    return PlanningData(status=status, time_taken=time_elapsed, nr_verts=nr_verts)

class PlanningData:

    def __init__(self, status, time_taken, nr_verts):
        self.status = status
        self.time_taken = time_taken
        self.nr_verts = nr_verts



