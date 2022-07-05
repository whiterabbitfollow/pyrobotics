
class Status:
    SUCCESS = "success"
    FAILURE = "failure"


class PlanningData:

    def __init__(self, status, time_taken):
        self.status = status
        self.time_taken = time_taken
