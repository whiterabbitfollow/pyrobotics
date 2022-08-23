import time
import numpy as np


def start_timer():
    time_s = time.time()
    time_elapsed = time.time() - time_s
    return time_s, time_elapsed


class Status:
    SUCCESS = "success"
    FAILURE = "failure"


def compile_planning_data(path, time_elapsed, meta_data):
    status = Status.SUCCESS if path.size else Status.FAILURE
    return PlanningData(status=status, time_taken=time_elapsed, meta_data=meta_data)


class PlanningData:

    def __init__(self, status, time_taken, meta_data):
        self.status = status
        self.time_taken = time_taken
        self.meta_data = meta_data


class PlanningProblem:

    def __init__(self, planner):
        self.planner = planner

    def solve(self, state_start, goal_region, max_planning_time=np.inf, min_planning_time=0):
        time_s, time_elapsed = start_timer()
        self.planner.clear()
        self.planner.initialize_planner(state_start, goal_region)
        while (
                self.planner.can_run() and
                (
                        (not self.planner.found_path and time_elapsed < max_planning_time)
                        or
                        (time_elapsed < min_planning_time)
                )
        ):
            self.planner.run()
            time_elapsed = time.time() - time_s
        path = self.planner.get_path()
        return path, compile_planning_data(path, time_elapsed, self.planner.get_planning_meta_data())

