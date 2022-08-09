
class Status:
    SUCCESS = "success"
    FAILURE = "failure"


class PlanningData:

    def __init__(self, status, time_taken, nr_verts):
        self.status = status
        self.time_taken = time_taken
        self.nr_verts = nr_verts
