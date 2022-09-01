import time
import numpy as np


def start_timer():
    time_s = time.time()
    time_elapsed = time.time() - time_s
    return time_s, time_elapsed


class SolutionStatus:
    SUCCESS = "success"
    FAILURE = "failure"


class PlanningData:

    def __init__(self, status, meta_data_problem, meta_data_planner):
        self.status = status
        self.meta_data_problem = meta_data_problem
        self.meta_data_planner = meta_data_planner


class PlanningProblem:

    def __init__(self, planner, debug=False):
        self.planner = planner
        self.debug = debug

    def set_planner(self, planner):
        self.planner = planner

    def solve(
            self,
            state_start,
            goal_region,
            max_planning_time=np.inf,
            min_planning_time=0,
            max_iters=np.inf,
            clear=True
    ):
        time_s, time_elapsed = start_timer()
        time_first_found = None
        if clear:
            self.planner.clear()
        self.planner.initialize_planner(state_start, goal_region)
        iter_cnt = 0
        while (
                self.planner.can_run() and
                (
                        (not self.planner.found_path and time_elapsed < max_planning_time)
                        or
                        (time_elapsed < min_planning_time)
                )
        ):
            self.planner.run()
            if self.debug:
                self.planner.debug(iter_cnt)
            time_elapsed = time.time() - time_s
            iter_cnt += 1
            if iter_cnt > max_iters:
                break
            if time_first_found is None and self.planner.found_path:
                time_first_found = time.time() - time_s
        path = self.planner.get_path()
        status = SolutionStatus.SUCCESS if path.size else SolutionStatus.FAILURE
        meta_data_problem = {
            "iter_cnt": iter_cnt, "time_elapsed": time_elapsed, "time_first_found": time_first_found or np.nan
        }
        data = PlanningData(
            status=status,
            meta_data_problem=meta_data_problem,
            meta_data_planner=self.planner.get_planning_meta_data()
        )
        return path, data


